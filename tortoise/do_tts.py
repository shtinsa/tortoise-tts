import argparse
import os

import torch
import torchaudio
import time
from api import TextToSpeech, MODELS_DIR, load_discrete_vocoder_diffuser
from utils.audio import load_voices

PROFILE=False

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--text', type=str, help='Text to speak.', default="The expressiveness of autoregressive transformers is literally nuts! I absolutely adore them.")
    parser.add_argument('--voice', type=str, help='Selects the voice to use for generation. See options in voices/ directory (and add your own!) '
                                                 'Use the & character to join two voices together. Use a comma to perform inference on multiple voices.', default='random')
    parser.add_argument('--preset', type=str, help='Which voice preset to use.', default='fast')
    parser.add_argument('--use_deepspeed', type=str, help='Which voice preset to use.', default=False)
    parser.add_argument('--kv_cache', type=bool, help='If you disable this please wait for a long a time to get the output', default=True)
    parser.add_argument('--half', type=bool, help="float16(half) precision inference if True it's faster and take less vram and ram", default=True)
    parser.add_argument('--output_path', type=str, help='Where to store outputs.', default='results/')
    parser.add_argument('--model_dir', type=str, help='Where to find pretrained model checkpoints. Tortoise automatically downloads these to .models, so this'
                                                      'should only be specified if you have custom checkpoints.', default=MODELS_DIR)
    parser.add_argument('--candidates', type=int, help='How many output candidates to produce per-voice.', default=1)
    parser.add_argument('--seed', type=int, help='Random seed which can be used to reproduce results.', default=None)
    parser.add_argument('--produce_debug_state', type=bool, help='Whether or not to produce debug_state.pth, which can aid in reproducing problems. Defaults to true.', default=True)
    parser.add_argument('--cvvp_amount', type=float, help='How much the CVVP model should influence the output.'
                                                          'Increasing this can in some cases reduce the likelihood of multiple speakers. Defaults to 0 (disabled)', default=.0)
    args = parser.parse_args()
    if torch.backends.mps.is_available():
        args.use_deepspeed = False
    os.makedirs(args.output_path, exist_ok=True)
    tts = TextToSpeech(models_dir=args.model_dir, use_deepspeed=args.use_deepspeed, kv_cache=args.kv_cache, half=args.half)

    selected_voices = args.voice.split(',')
    for k, selected_voice in enumerate(selected_voices):
        if '&' in selected_voice:
            voice_sel = selected_voice.split('&')
        else:
            voice_sel = [selected_voice]
        voice_samples, conditioning_latents = load_voices(voice_sel)
        backend = "PYTORCH"
        diffuser = None
        # print("args", args)
        # print("args.preset", args.preset)
        if 'USE_TVM_MODEL'  in os.environ:
            backend = "TVM"
            diffuser = load_discrete_vocoder_diffuser(desired_diffusion_steps=30,
                                                      cond_free=False, cond_free_k=2.0)
            auto_conditioning, diffusion_conditioning, auto_conds, _ = tts.get_conditioning_latents(voice_samples, return_mels=True)
            del conditioning_latents
            conditioning_latents = (auto_conditioning, diffusion_conditioning, auto_conds)
            del voice_samples
            voice_samples = None

        # print("diffuser->", diffuser)
        start_tm = time.time()
        if PROFILE == False:
            gen, dbg_state = tts.tts_with_preset(args.text, diffuser=diffuser, k=args.candidates, voice_samples=voice_samples,
                                                 conditioning_latents=conditioning_latents,
                                      preset=args.preset, use_deterministic_seed=args.seed, return_deterministic_state=True,
                                      cvvp_amount=args.cvvp_amount)
        else:
            from torch.profiler import profile, ProfilerActivity
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=False, with_modules=False, with_stack=True) as prof:
                gen, dbg_state = tts.tts_with_preset(args.text, diffuser=diffuser, k=args.candidates, voice_samples=voice_samples,
                                                     conditioning_latents=conditioning_latents,
                                        preset=args.preset, use_deterministic_seed=args.seed, return_deterministic_state=True, cvvp_amount=args.cvvp_amount)
            os.makedirs("traces",exist_ok=True)
            export_traces = f'./traces/trace_neonjb_{backend}.json'
            export_logs = f"./traces/trace_neonjb_{backend}.txt"

            prof.export_chrome_trace(export_traces)
            with open(export_logs, 'w') as fle:
                fle.write("{}".format(prof.key_averages(group_by_stack_n=5).table(sort_by="self_cuda_time_total")))

        print(f"{backend} Infer time:", time.time() - start_tm, ' sec.')
        if isinstance(gen, list):
            for j, g in enumerate(gen):
                torchaudio.save(os.path.join(args.output_path, f'{selected_voice}_{k}_{j}.wav'), g.squeeze(0).cpu(), 24000)
        else:
            torchaudio.save(os.path.join(args.output_path, f'{selected_voice}_{k}.wav'), gen.squeeze(0).cpu(), 24000)

        if args.produce_debug_state:
            os.makedirs('debug_states', exist_ok=True)
            torch.save(dbg_state, f'debug_states/do_tts_debug_{selected_voice}.pth')

