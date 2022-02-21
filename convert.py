import os
import librosa
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.signal import savgol_filter


import preprocess
from model_tf import Generator, Discriminator


class CycleGANConvert:
    def __init__(self,
                 logf0s_normalization,
                 mcep_normalization,
                 val_a_dir,
                 val_b_dir,
                 conv_a_dir,
                 conv_b_dir,
                 checkpoint,
                 use_cpu=False):
        self.use_cpu = use_cpu
        self.validation_A_dir = val_a_dir
        self.validation_B_dir = val_b_dir
        self.output_A_dir = conv_a_dir
        self.output_B_dir = conv_b_dir
        os.makedirs(self.output_A_dir, exist_ok=True)
        os.makedirs(self.output_B_dir, exist_ok=True)

        self.device = torch.device(
            'cuda' if torch.cuda.is_available() and not use_cpu else 'cpu')

        self.generator_A2B = Generator().to(self.device)
        self.generator_B2A = Generator().to(self.device)
        self.discriminator_A = Discriminator().to(self.device)
        self.discriminator_B = Discriminator().to(self.device)

        g_params = list(self.generator_A2B.parameters()) + \
                   list(self.generator_B2A.parameters())
        d_params = list(self.discriminator_A.parameters()) + \
                   list(self.discriminator_B.parameters())

        # Initial learning rates
        self.generator_lr = 2e-4  # 0.0002
        self.discriminator_lr = 1e-4  # 0.0001

        logf0s_normalization = np.load(logf0s_normalization)
        self.log_f0s_mean_A = logf0s_normalization['mean_A']
        self.log_f0s_std_A = logf0s_normalization['std_A']
        self.log_f0s_mean_B = logf0s_normalization['mean_B']
        self.log_f0s_std_B = logf0s_normalization['std_B']

        self.generator_optimizer = torch.optim.Adam(
            g_params, lr=self.generator_lr, betas=(0.5, 0.999))
        self.discriminator_optimizer = torch.optim.Adam(
            d_params, lr=self.discriminator_lr, betas=(0.5, 0.999))

        mcep_normalization = np.load(mcep_normalization)
        self.coded_sps_A_mean = mcep_normalization['mean_A']
        self.coded_sps_A_std = mcep_normalization['std_A']
        self.coded_sps_B_mean = mcep_normalization['mean_B']
        self.coded_sps_B_std = mcep_normalization['std_B']

        self.loadModel(checkpoint)

    def loadModel(self, PATH):
        checkPoint = torch.load(PATH)
        self.generator_A2B.load_state_dict(
            state_dict=checkPoint['model_genA2B_state_dict'])
        self.generator_B2A.load_state_dict(
            state_dict=checkPoint['model_genB2A_state_dict'])
        self.discriminator_A.load_state_dict(
            state_dict=checkPoint['model_discriminatorA'])
        self.discriminator_B.load_state_dict(
            state_dict=checkPoint['model_discriminatorB'])
        self.generator_optimizer.load_state_dict(
            state_dict=checkPoint['generator_optimizer'])
        self.discriminator_optimizer.load_state_dict(
            state_dict=checkPoint['discriminator_optimizer'])
        self.generator_loss_store = checkPoint['generator_loss_store']
        self.discriminator_loss_store = checkPoint['discriminator_loss_store']

    def validation_for_A_dir(self):
        num_mcep = 35
        sampling_rate = 16000
        frame_period = 5.0
        validation_A_dir = self.validation_A_dir
        output_A_dir = self.output_A_dir

        print("Generating Validation Data B from A...")
        for file in os.listdir(validation_A_dir):
            filePath = os.path.join(validation_A_dir, file)
            wav, _ = librosa.load(filePath, sr=sampling_rate, mono=True)
            wav = preprocess.wav_padding(wav=wav,
                                         sr=sampling_rate,
                                         frame_period=frame_period,
                                         multiple=4)
            f0, timeaxis, sp, ap = preprocess.world_decompose(
                wav=wav, fs=sampling_rate, frame_period=frame_period)
            logf0 = np.log(f0 + 1)
            logf0_norm = (logf0 - self.log_f0s_mean_A) / self.log_f0s_std_A
            coded_sp = preprocess.world_encode_spectral_envelop(
                sp=sp, fs=sampling_rate, dim=num_mcep)
            coded_sp_transposed = coded_sp.T
            coded_sp_norm = (coded_sp_transposed -
                             self.coded_sps_A_mean) / self.coded_sps_A_std
            coded_sp_norm = np.array([coded_sp_norm])
            logf0_norm = logf0_norm.reshape(1, 1, -1)
            gen_input = np.concatenate((coded_sp_norm, logf0_norm), axis=1)

            if torch.cuda.is_available():
                gen_input = torch.from_numpy(gen_input).cuda().float()
            else:
                gen_input = torch.from_numpy(gen_input).float()

            gen_input_conv = self.generator_A2B(gen_input)
            gen_input_conv = gen_input_conv.cpu().detach().numpy()
            gen_input_conv = np.squeeze(gen_input_conv)
            coded_sp_converted_norm = gen_input_conv[:-1]
            logf0_converted_norm = gen_input_conv[-1]

            coded_sp_converted = coded_sp_converted_norm * \
                                 self.coded_sps_B_std + self.coded_sps_B_mean
            coded_sp_converted = coded_sp_converted.T
            coded_sp_converted = np.ascontiguousarray(coded_sp_converted)
            decoded_sp_converted = preprocess.world_decode_spectral_envelop(
                coded_sp=coded_sp_converted, fs=sampling_rate)
            logf0_converted = \
                logf0_converted_norm * self.log_f0s_std_B + self.log_f0s_mean_B
            f0_converted = np.exp(logf0_converted) - 1
            f0_converted = f0_converted.clip(min=0).astype(np.float64)
            # plt.plot(f0_converted, color="blue", linestyle='dashed')
            f0_converted = savgol_filter(f0_converted, 11, 2)
            f0_converted *= np.not_equal(f0, 0)
            # plt.plot(f0_converted, color="red")
            # plt.plot(f0, color="green")
            # plt.show()
            wav_transformed = preprocess.world_speech_synthesis(f0=f0_converted,
                                                                decoded_sp=sp,
                                                                ap=ap,
                                                                fs=sampling_rate,
                                                                frame_period=frame_period)
            librosa.output.write_wav(path=os.path.join(output_A_dir, os.path.basename(file)),
                                     y=wav_transformed,
                                     sr=sampling_rate)

    def validation_for_B_dir(self):
        num_mcep = 36
        sampling_rate = 16000
        frame_period = 5.0
        validation_B_dir = self.validation_B_dir
        output_B_dir = self.output_B_dir

        print("Generating Validation Data A from B...")
        for file in os.listdir(validation_B_dir):
            filePath = os.path.join(validation_B_dir, file)
            wav, _ = librosa.load(filePath, sr=sampling_rate, mono=True)
            wav = preprocess.wav_padding(wav=wav,
                                         sr=sampling_rate,
                                         frame_period=frame_period,
                                         multiple=4)
            f0, timeaxis, sp, ap = preprocess.world_decompose(
                wav=wav, fs=sampling_rate, frame_period=frame_period)
            f0_converted = preprocess.pitch_conversion(f0=f0,
                                                       mean_log_src=self.log_f0s_mean_B,
                                                       std_log_src=self.log_f0s_std_B,
                                                       mean_log_target=self.log_f0s_mean_A,
                                                       std_log_target=self.log_f0s_std_A)
            coded_sp = preprocess.world_encode_spectral_envelop(
                sp=sp, fs=sampling_rate, dim=num_mcep)
            coded_sp_transposed = coded_sp.T
            coded_sp_norm = (coded_sp_transposed -
                             self.coded_sps_B_mean) / self.coded_sps_B_std
            coded_sp_norm = np.array([coded_sp_norm])

            if torch.cuda.is_available() and not self.use_cpu:
                coded_sp_norm = torch.from_numpy(coded_sp_norm).cuda().float()
            else:
                coded_sp_norm = torch.from_numpy(coded_sp_norm).float()

            coded_sp_converted_norm = self.generator_B2A(coded_sp_norm)
            coded_sp_converted_norm = coded_sp_converted_norm.cpu().detach().numpy()
            coded_sp_converted_norm = np.squeeze(coded_sp_converted_norm)
            coded_sp_converted = coded_sp_converted_norm * \
                                 self.coded_sps_A_std + self.coded_sps_A_mean
            coded_sp_converted = coded_sp_converted.T
            coded_sp_converted = np.ascontiguousarray(coded_sp_converted)
            decoded_sp_converted = preprocess.world_decode_spectral_envelop(
                coded_sp=coded_sp_converted, fs=sampling_rate)
            wav_transformed = preprocess.world_speech_synthesis(f0=f0_converted,
                                                                decoded_sp=decoded_sp_converted,
                                                                ap=ap,
                                                                fs=sampling_rate,
                                                                frame_period=frame_period)
            librosa.output.write_wav(path=os.path.join(output_B_dir, os.path.basename(file)),
                                     y=wav_transformed,
                                     sr=sampling_rate)


if __name__ == "__main__":
    logf0s_normalization = './cache/logf0s_normalization.npz'
    mcep_normalization = './cache/mcep_normalization.npz'
    val_a_dir = '/shared_data/data/nfs/emo_conversion/datasets/neu2hap_personal/val/neu'
    val_b_dir = '/shared_data/data/nfs/emo_conversion/datasets/neu2hap_personal/train/hap'
    conv_a_dir = './converted_sound/neu_f0'
    conv_b_dir = './converted_sound/hap_f0'
    checkpoint = './model_checkpoint/_CycleGAN_CheckPoint'

    converter = CycleGANConvert(
        logf0s_normalization=logf0s_normalization,
        mcep_normalization=mcep_normalization,
        val_a_dir=val_a_dir,
        val_b_dir=val_b_dir,
        conv_a_dir=conv_a_dir,
        conv_b_dir=conv_b_dir,
        checkpoint=checkpoint,
        use_cpu=False
    )
    converter.validation_for_A_dir()
    converter.validation_for_B_dir()
