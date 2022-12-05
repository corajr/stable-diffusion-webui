import bisect
import copy
import cv2
import librosa
import numpy as np
import srt
import torch

import modules.scripts as scripts
import gradio as gr

from modules import images
from modules.processing import process_images, Processed
from modules.shared import opts, cmd_opts, state
import modules.sd_samplers

def gen_latents(audio_filename, polar=True, step_size=64):
    y, sr = librosa.load(audio_filename, mono=False)  # (2, t)
    cqt = torch.tensor(librosa.cqt(y, sr=sr, n_bins=64)) # (2, 64, t)
    cqt = torch.nn.functional.pad(cqt, (0, 63), "constant", 0)
    ts = librosa.frames_to_time(np.arange(cqt.shape[-1]), sr=sr)

    cqt = cqt.permute(2, 0, 1).unfold(0, 64, 1)  # (t, 2, 64, 64)
    cqt = torch.cat((torch.real(cqt), torch.imag(cqt)), 1)  # (t, 4, 64, 64)
    batch_norm = torch.nn.BatchNorm2d(4, affine=False)
    cqt = batch_norm(cqt)

    if polar:
        cqt = torch.nn.Sigmoid()(cqt)
        for i, frame in enumerate(cqt): 
            img = frame.permute(1, 2, 0).numpy()
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            polar = cv2.warpPolar(img, center=(31, 31), maxRadius=32, dsize=(64, 64), flags=cv2.WARP_INVERSE_MAP)
            cqt[i, ...] = torch.tensor(polar).permute(2, 0, 1)
        cqt = torch.special.logit(cqt, eps=1e-5)

    return cqt[::step_size, ...], ts[::step_size]


def lerp(prompt1, prompt2, alpha):
    return f"{prompt1}:{1.0 - alpha} AND {prompt2}:{alpha}"


def subs_to_prompts(subs, n, step_size=1, suffix=""):
    all_prompts = []
    subs_i = 0
    m = len(subs)
    for i in range(n):
        t = i * (step_size * (512 / 22050))

        while t >= subs[subs_i].end.total_seconds() and subs_i < m - 1:
            subs_i = subs_i + 1

        all_prompts.append(subs[subs_i].content + (", " + suffix if suffix else ""))
        i += 1
    return all_prompts


def tweens(subs, n, step_size=64, blank_prompt="golden starry sky", suffix=""):
    centers = [(-1.0, blank_prompt)]
    for sub in subs:
        center = sub.start + ((sub.end - sub.start) / 2)
        centers.append((center.total_seconds(), sub.content))
    m = len(centers)
    t_final = n * (step_size * (512 / 22050))

    prompts = []
    for i in range(n):
        t = i * (step_size * (512 / 22050))
        center_i = bisect.bisect_left(centers, t, key=lambda x: x[0])
        if center_i >= m:
            cur, nex = centers[m - 1], (t_final + 1.0, blank_prompt)
        else:
            cur, nex = centers[center_i - 1], centers[center_i]
        t1, t2 = cur[0], nex[0]
        dur = t2 - t1
        alpha = (t - t1) / dur

        prompt1, prompt2 = cur[1], nex[1]
        prompts.append(lerp(prompt1 + ", " + suffix, prompt2 + ", " + suffix, alpha))
        
    return prompts


class Script(scripts.Script):
    def title(self):
        return "Audio latents"

    def ui(self, is_img2img):
        audio_file = gr.Audio(label="Upload audio file", type="filepath")
        srt_file = gr.File(label="Upload subtitle file (if not supplied, will use same prompt throughout)", type="bytes")
        seed_strength = gr.Slider(label="Seed strength", minimum=0.0, maximum=1.0, step=0.01, value=0.6)
        step_size = gr.Slider(minimum=1, maximum=2048, step=1, label="Number of CQT frames between images (1 frame = 23.2 ms)", value=5)
        suffix = gr.Textbox(label="Suffix to append to prompt", value="beautiful and highly detailed fantasy concept art, circular painting by Greg Rutkowski and Leo and Diane Dillon")
        only_changes = gr.Checkbox(label="Only generate images when the prompt changes", value=False)
        tween_text = gr.Checkbox(label="Whether to interpolate between prompts", value=False)
        polar = gr.Checkbox(label="Whether to use polar mapping", value=True)
        offset = gr.Slider(minimum=0, maximum=10000, step=1, label="Frame at which to begin processing", value=0)
        limit = gr.Slider(minimum=0, maximum=100, step=1, label="Limit of frames to output (0 is unlimited)", value=0)

        self.infotext_fields = [
            (seed_strength, "Seed strength"),
            (audio_file, "Audio filepath"),
            (srt_file, "SRT filepath"),
            (polar, "Polar mapping"),
            (suffix, "Suffix"),
            (tween_text, "Tween text prompts")
        ]

        return [audio_file, srt_file, seed_strength, step_size, suffix, only_changes, limit, polar, tween_text, offset]


    def run(self, p, audio_file, srt_file, seed_strength, step_size, suffix, only_changes, limit, polar, tween_text, offset):
        modules.processing.fix_seed(p)
        latents, ts = gen_latents(audio_file, polar=polar, step_size=step_size)

        if srt_file:
            subs = list(srt.parse(srt_file.decode("utf-8", errors="ignore")))

            if tween_text:
                all_prompts = tweens(subs, latents.shape[0], step_size, suffix=suffix)
            else:
                all_prompts = subs_to_prompts(subs, latents.shape[0], step_size, suffix=suffix)
        else:
            all_prompts = [p.prompt] * latents.shape[0]

        if only_changes:
            prev_prompt = None
            indices = []
            reduced_prompts = []
            reduced_ts = []
            for i, prompt in enumerate(all_prompts):
                if prompt != prev_prompt:
                    indices.append(i)
                    reduced_ts.append(ts[i])
                    reduced_prompts.append(prompt)
                    prev_prompt = prompt
            all_prompts = reduced_prompts
            ts = reduced_ts
            latents = latents[indices, ...]

        if limit == 0:
            limit = latents.shape[0]

        all_prompts = all_prompts[offset:offset + limit]
        latents = latents[offset:offset + limit, ...]
        ts = ts[offset:offset + limit]

        state.job_count = latents.shape[0]

        images = []
        infotexts = []
        for t, prompt, latent in zip(ts, all_prompts, latents):
            state.job = f"{state.job_no + 1} out of {state.job_count}"

            copy_p = copy.copy(p)
            copy_p.prompt = prompt
            copy_p.seed_strength = seed_strength
            copy_p.init_latent = latent.unsqueeze(0)

            infotext_fields = [
                (seed_strength, "Seed strength"),
                (audio_file, "Audio filepath"),
                (t, "Audio timestamp"),
                (polar, "Polar mapping"),
                (suffix, "Suffix"),
                (tween_text, "Tween text prompts")
            ]
            copy_p.extra_generation_params = dict((y,x) for x, y in infotext_fields)

            proc = process_images(copy_p)
            images += proc.images
            infotexts.extend(proc.infotexts)

        return Processed(p, images, p.seed, "",
                         all_prompts=all_prompts,
                         all_seeds=[p.seed] * len(all_prompts),
                         infotexts=infotexts)
