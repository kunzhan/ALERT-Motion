# This code is based on https://github.com/openai/guided-diffusion
"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""
from utils.fixseed import fixseed
import os
import numpy as np
import torch
# from utils.parser_util import generate_args
from utils.model_utils import MDM_Model, load_model_wo_clip
from utils import dist_util
from model.cfg_sampler import ClassifierFreeSampleModel
from data_loaders.get_data import get_dataset_loader
from data_loaders.humanml.scripts.motion_process import recover_from_ric
import data_loaders.humanml.utils.paramUtil as paramUtil
from data_loaders.humanml.utils.plot_script import plot_3d_motion
import shutil
from data_loaders.tensors import collate





class Generate() :

    def __init__(self, device):
        #nothing to be initialized
        self.seed = 10
        # self.output_dir = output_dir
        self.dataset = 'humanml'
        self.input_text = None
        self.action_file = None
        self.action_name = None
        self.device = device
        self.num_samples = 1
        self.batch_size = 64
        self.num_repetitions = 1
        self.model_path = "/home/mhl/code/baseline/target_model/mdm/save/humanml_trans_enc_512/model000475000.pt"
        self.motion_length = 6.0
        self.guidance_param = 2.5
        self.unconstrained = False
        # self.num_timesteps = num_timesteps


        dist_util.setup_dist(self.device)
        self.max_frames = 196 if self.dataset in ['kit', 'humanml'] else 60
        self.fps = 12.5 if self.dataset == 'kit' else 20
        self.n_frames = min(self.max_frames, int(self.motion_length*self.fps))
        # this block must be called BEFORE the dataset is loaded

        self.batch_size = self.num_samples  # Sampling a single batch from the testset, with exactly self.num_samples

        # print('Loading dataset...')
        data = self.load_dataset()
        self.total_num_samples = self.num_samples * self.num_repetitions

        # print("Creating model and diffusion...")
        mdm = MDM_Model()
        model, diffusion = mdm.create_model_and_diffusion(data=data)


        # print(f"Loading checkpoints from [{self.model_path}]...")
        state_dict = torch.load(self.model_path, map_location='cpu')
        load_model_wo_clip(model, state_dict)

        if self.guidance_param != 1:
            model = ClassifierFreeSampleModel(model)   # wrapping model with the classifier-free sampler
        model.to(dist_util.dev())
        model.eval()  # disable random masking
        self.model = model
        self.diffusion = diffusion
        self.data = data


    def generate_motion(self, text_prompt, output_dir, num_timesteps, render):
        self.diffusion.num_timesteps = num_timesteps
        fixseed(self.seed)
        out_path = output_dir
        name = os.path.basename(os.path.dirname(self.model_path))
        niter = os.path.basename(self.model_path).replace('model', '').replace('.pt', '')

        if text_prompt != '':
            texts = [text_prompt]
            self.num_samples = 1
        elif self.input_text != '':
            assert os.path.exists(self.input_text)
            with open(self.input_text, 'r') as fr:
                texts = fr.readlines()
            texts = [s.replace('\n', '') for s in texts]
            self.num_samples = len(texts)
        elif self.action_name:
            action_text = [self.action_name]
            self.num_samples = 1
        elif self.action_file != '':
            assert os.path.exists(self.action_file)
            with open(self.action_file, 'r') as fr:
                action_text = fr.readlines()
            action_text = [s.replace('\n', '') for s in action_text]
            self.num_samples = len(action_text)

        assert self.num_samples <= self.batch_size, \
            f'Please either increase batch_size({self.batch_size}) or reduce num_samples({self.num_samples})'
        is_using_data = not any([self.input_text, text_prompt, self.action_file, self.action_name])
        # args = generate_args()
        # fixseed(self.seed)
        # out_path = output_dir
        # name = os.path.basename(os.path.dirname(self.model_path))
        # niter = os.path.basename(self.model_path).replace('model', '').replace('.pt', '')
        # self.max_frames = 196 if self.dataset in ['kit', 'humanml'] else 60
        # self.fps = 12.5 if self.dataset == 'kit' else 20
        # self.n_frames = min(self.max_frames, int(self.motion_length*self.fps))
        # is_using_data = not any([self.input_text, text_prompt, self.action_file, self.action_name])
        # dist_util.setup_dist(self.device)
        # # if out_path == '':
        #     # out_path = os.path.join(os.path.dirname(self.model_path),
        #     #                         'samples_{}_{}_seed{}'.format(name, niter, self.seed))
        #     # if text_prompt != '':
        #     #     out_path += '_' + text_prompt.replace(' ', '_').replace('.', '')
        #     # elif self.input_text != '':
        #     #     out_path += '_' + os.path.basename(self.input_text).replace('.txt', '').replace(' ', '_').replace('.', '')

        # # this block must be called BEFORE the dataset is loaded
        # if text_prompt != '':
        #     texts = [text_prompt]
        #     self.num_samples = 1
        # elif self.input_text != '':
        #     assert os.path.exists(self.input_text)
        #     with open(self.input_text, 'r') as fr:
        #         texts = fr.readlines()
        #     texts = [s.replace('\n', '') for s in texts]
        #     self.num_samples = len(texts)
        # elif self.action_name:
        #     action_text = [self.action_name]
        #     self.num_samples = 1
        # elif self.action_file != '':
        #     assert os.path.exists(self.action_file)
        #     with open(self.action_file, 'r') as fr:
        #         action_text = fr.readlines()
        #     action_text = [s.replace('\n', '') for s in action_text]
        #     self.num_samples = len(action_text)

        # assert self.num_samples <= self.batch_size, \
        #     f'Please either increase batch_size({self.batch_size}) or reduce num_samples({self.num_samples})'
        # # So why do we need this check? In order to protect GPU from a memory overload in the following line.
        # # If your GPU can handle batch size larger then default, you can specify it through --batch_size flag.
        # # If it doesn't, and you still want to sample more prompts, run this script with different seeds
        # # (specify through the --seed flag)
        # self.batch_size = self.num_samples  # Sampling a single batch from the testset, with exactly self.num_samples

        # # print('Loading dataset...')
        # data = self.load_dataset(self.max_frames, self.n_frames)
        # self.total_num_samples = self.num_samples * self.num_repetitions

        # # print("Creating model and diffusion...")
        # mdm = MDM_Model()
        # model, diffusion = mdm.create_model_and_diffusion(data=data)
        # diffusion.num_timesteps = self.num_timesteps

        # # print(f"Loading checkpoints from [{self.model_path}]...")
        # state_dict = torch.load(self.model_path, map_location='cpu')
        # load_model_wo_clip(model, state_dict)

        # if self.guidance_param != 1:
        #     model = ClassifierFreeSampleModel(model)   # wrapping model with the classifier-free sampler
        # model.to(dist_util.dev())
        # model.eval()  # disable random masking

        if is_using_data:
            iterator = iter(self.data)
            _, model_kwargs = next(iterator)
        else:
            collate_self = [{'inp': torch.zeros(self.n_frames), 'tokens': None, 'lengths': self.n_frames}] * self.num_samples
            is_t2m = any([self.input_text, text_prompt])
            if is_t2m:
                # t2m
                collate_self = [dict(arg, text=txt) for arg, txt in zip(collate_self, texts)]
            else:
                # a2m
                action = self.data.dataset.action_name_to_action(action_text)
                collate_self = [dict(arg, action=one_action, action_text=one_action_text) for
                                arg, one_action, one_action_text in zip(collate_self, action, action_text)]
            _, model_kwargs = collate(collate_self)

        all_motions = []
        all_lengths = []
        all_text = []

        for rep_i in range(self.num_repetitions):
            print(f'### Sampling [repetitions #{rep_i}]')

            # add CFG scale to batch
            if self.guidance_param != 1:
                model_kwargs['y']['scale'] = torch.ones(self.batch_size, device=dist_util.dev()) * self.guidance_param

            sample_fn = self.diffusion.p_sample_loop# ddim_sample_loop

            sample = sample_fn(
                self.model,
                # (self.batch_size, model.njoints, model.nfeats, self.n_frames),  # BUG FIX - this one caused a mismatch between training and inference
                (self.batch_size, self.model.njoints, self.model.nfeats, self.max_frames),  # BUG FIX
                clip_denoised=False,
                model_kwargs=model_kwargs,
                skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
                init_image=None,
                progress=True,
                dump_steps=None,
                noise=None,
                const_noise=False,
            )

            npy_path = out_path
            if not render:
                np.save(npy_path, sample.cpu().numpy())

            # Recover XYZ *positions* from HumanML3D vector representation
            if self.model.data_rep == 'hml_vec':
                n_joints = 22 if sample.shape[1] == 263 else 21
                sample = self.data.dataset.t2m_dataset.inv_transform(sample.cpu().permute(0, 2, 3, 1)).float()
                sample = recover_from_ric(sample, n_joints)
                sample = sample.view(-1, *sample.shape[2:]).permute(0, 2, 3, 1)

            rot2xyz_pose_rep = 'xyz' if self.model.data_rep in ['xyz', 'hml_vec'] else self.model.data_rep
            rot2xyz_mask = None if rot2xyz_pose_rep == 'xyz' else model_kwargs['y']['mask'].reshape(self.batch_size, self.n_frames).bool()
            sample = self.model.rot2xyz(x=sample, mask=rot2xyz_mask, pose_rep=rot2xyz_pose_rep, glob=True, translation=True,
                                jointstype='smpl', vertstrans=True, betas=None, beta=0, glob_rot=None,
                                get_rotations_back=False)

            if self.unconstrained:
                all_text += ['unconstrained'] * self.num_samples
            else:
                text_key = 'text' if 'text' in model_kwargs['y'] else 'action_text'
                all_text += model_kwargs['y'][text_key]

            if render:
                np.save(npy_path, sample.cpu().numpy())
            all_motions.append(sample.cpu().numpy())
            all_lengths.append(model_kwargs['y']['lengths'].cpu().numpy())

            print(f"created {len(all_motions) * self.batch_size} samples")


        all_motions = np.concatenate(all_motions, axis=0)
        all_motions = all_motions[:self.total_num_samples]  # [bs, njoints, 6, seqlen]
        all_text = all_text[:self.total_num_samples]
        all_lengths = np.concatenate(all_lengths, axis=0)[:self.total_num_samples]

        # if os.path.exists(out_path):
        #     shutil.rmtree(out_path)
        # os.makedirs(out_path)

        # npy_path = out_path# os.path.join(out_path, 'results.npy')
        # print(f"saving results file to [{npy_path}]")
        # np.save(npy_path,
        #         {'motion': all_motions, 'text': all_text, 'lengths': all_lengths,
        #         'num_samples': self.num_samples, 'num_repetitions': self.num_repetitions})
        with open(npy_path.replace('.npy', '.txt'), 'w') as fw:
            fw.write('\n'.join(all_text))
        with open(npy_path.replace('.npy', '_len.txt'), 'w') as fw:
            fw.write('\n'.join([str(l) for l in all_lengths]))

        # print(f"saving visualizations to [{out_path}]...")
        skeleton = paramUtil.kit_kinematic_chain if self.dataset == 'kit' else paramUtil.t2m_kinematic_chain

        sample_files = []
        num_samples_in_out_file = 7

        sample_print_template, row_print_template, all_print_template, \
        sample_file_template, row_file_template, all_file_template = self.construct_template_variables(self.unconstrained)

        for sample_i in range(self.num_samples):
            # rep_files = []
            for rep_i in range(self.num_repetitions):
                caption = all_text[rep_i*self.batch_size + sample_i]
                length = all_lengths[rep_i*self.batch_size + sample_i]
                motion = all_motions[rep_i*self.batch_size + sample_i].transpose(2, 0, 1)[:length]
                # save_file = sample_file_template.format(sample_i, rep_i)
                # print(sample_print_template.format(caption, sample_i, rep_i, save_file))
                portion = os.path.splitext(out_path)
                animation_save_path = portion[0]+".mp4"# os.path.join(out_path, save_file)
                plot_3d_motion(animation_save_path, skeleton, motion, dataset=self.dataset, title=caption)
                # Credit for visualization: https://github.com/EricGuo5513/text-to-motion
                # rep_files.append(animation_save_path)

            # sample_files = self.save_multiple_samples(self, out_path,
            #                                     row_print_template, all_print_template, row_file_template, all_file_template,
            #                                     caption, num_samples_in_out_file, rep_files, sample_files, sample_i)

        abs_path = os.path.abspath(out_path)
        print(f'[Done] Results are at [{abs_path}]')


    # def save_multiple_samples(self, out_path, row_print_template, all_print_template, row_file_template, all_file_template,
    #                         caption, num_samples_in_out_file, rep_files, sample_files, sample_i):
    #     all_rep_save_file = row_file_template.format(sample_i)
    #     all_rep_save_path = os.path.join(out_path, all_rep_save_file)
    #     ffmpeg_rep_files = [f' -i {f} ' for f in rep_files]
    #     hstack_self = f' -filter_complex hstack=inputs={self.num_repetitions}' if self.num_repetitions > 1 else ''
    #     ffmpeg_rep_cmd = f'ffmpeg -y -loglevel warning ' + ''.join(ffmpeg_rep_files) + f'{hstack_self} {all_rep_save_path}'
    #     os.system(ffmpeg_rep_cmd)
    #     print(row_print_template.format(caption, sample_i, all_rep_save_file))
    #     sample_files.append(all_rep_save_path)
    #     if (sample_i + 1) % num_samples_in_out_file == 0 or sample_i + 1 == self.num_samples:
    #         # all_sample_save_file =  f'samples_{(sample_i - len(sample_files) + 1):02d}_to_{sample_i:02d}.mp4'
    #         all_sample_save_file = all_file_template.format(sample_i - len(sample_files) + 1, sample_i)
    #         all_sample_save_path = os.path.join(out_path, all_sample_save_file)
    #         print(all_print_template.format(sample_i - len(sample_files) + 1, sample_i, all_sample_save_file))
    #         ffmpeg_rep_files = [f' -i {f} ' for f in sample_files]
    #         vstack_self = f' -filter_complex vstack=inputs={len(sample_files)}' if len(sample_files) > 1 else ''
    #         ffmpeg_rep_cmd = f'ffmpeg -y -loglevel warning ' + ''.join(
    #             ffmpeg_rep_files) + f'{vstack_self} {all_sample_save_path}'
    #         os.system(ffmpeg_rep_cmd)
    #         sample_files = []
    #     return sample_files


    def construct_template_variables(self, unconstrained):
        row_file_template = 'sample{:02d}.mp4'
        all_file_template = 'samples_{:02d}_to_{:02d}.mp4'
        if unconstrained:
            sample_file_template = 'row{:02d}_col{:02d}.mp4'
            sample_print_template = '[{} row #{:02d} column #{:02d} | -> {}]'
            row_file_template = row_file_template.replace('sample', 'row')
            row_print_template = '[{} row #{:02d} | all columns | -> {}]'
            all_file_template = all_file_template.replace('samples', 'rows')
            all_print_template = '[rows {:02d} to {:02d} | -> {}]'
        else:
            sample_file_template = 'sample{:02d}_rep{:02d}.mp4'
            sample_print_template = '["{}" ({:02d}) | Rep #{:02d} | -> {}]'
            row_print_template = '[ "{}" ({:02d}) | all repetitions | -> {}]'
            all_print_template = '[samples {:02d} to {:02d} | all repetitions | -> {}]'

        return sample_print_template, row_print_template, all_print_template, \
            sample_file_template, row_file_template, all_file_template


    def load_dataset(self):
        data = get_dataset_loader(name=self.dataset,
                                batch_size=self.batch_size,
                                num_frames=self.max_frames,
                                split='test',
                                hml_mode='text_only')
        if self.dataset in ['kit', 'humanml']:
            data.dataset.t2m_dataset.fixed_length = self.n_frames
        return data


# if __name__ == "__main__":
#     main()
