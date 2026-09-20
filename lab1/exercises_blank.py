# Exercises for laboratory work


# Import of modules
import numpy as np
from scipy.fftpack import dct


def split_meta_line(line, delimiter=' '):
    # First you need to prepare split_meta_line function for meta line parsing
    # line format is "Speaker_ID Gender Path"
    
    """
    :param line: lines of metadata
    :param delimiter: delimeter
    :return: speaker_id: speaker IDs: gender: gender: file_path: path to file
    """

    ###########################################################
    # .strip() убирает невидимый символ переноса строки (\n) в конце
    # .split(delimiter) режет строку на части по разделителю
    speaker_id, gender, file_path = line.strip().split(delimiter)

    ###########################################################

    return speaker_id, gender, file_path

def preemphasis(signal, pre_emphasis=-0.97):
    #Here you need to preemphasis input signal with pre_emphasis coeffitient

    """
    :param signal: input signal
    :param pre_emphasis: preemphasis coeffitient
    :return: emphasized_signal: signal after pre-emphasis procedure
    """

    ###########################################################
    # y[t] = x[t] - A * x[t-1], где A - коэффициент альфа,
    # Для фильтра высоких частот альфа делать ОТРИЦАТЕЛЬНЫМ
    signal = np.asarray(signal)
    emphasized_signal = np.zeros_like(signal)
    
    # Первый отсчет y(0) оставляем как x(0), так как x(-1) не существует
    emphasized_signal[0] = signal[0]
    
    # Строго по формуле из лабораторной: y(n) = x(n) + alpha * x(n-1)
    emphasized_signal[1:] = signal[1:] + pre_emphasis * signal[:-1]

    ###########################################################

    return emphasized_signal

def framing(emphasized_signal, sample_rate=16000, frame_size=0.025, frame_stride=0.01):
    # Here you need to perform framing of the input signal emphasized_signal with sample rate sample_rate
    # Please use hamming windowing
    
    """
    :param emphasized_signal: signal after pre-emphasis procedure
    :param sample_rate: signal sampling rate
    :param frame_size: sliding window size
    :param frame_stride: step
    :return: frames: output matrix [nframes x sample_rate*frame_size]
    """

    frame_length, frame_step = frame_size * sample_rate, frame_stride * sample_rate # convert from seconds to samples
    signal_length = len(emphasized_signal)
    frame_length = int(round(frame_length))
    frame_step = int(round(frame_step))
    num_frames = int(
        np.ceil(float(np.abs(signal_length - frame_length)) / frame_step)) # make sure that we have at least 1 frame

    pad_signal_length = num_frames * frame_step + frame_length
    z = np.zeros((pad_signal_length - signal_length))
    pad_signal = np.append(emphasized_signal, z) # pad Signal to make sure that all frames have equal number of samples without
                                                 # truncating any samples from the original signal

    ###########################################################
    # 1. Генерируем матрицу индексов для всех фреймов сразу
    # Строка arange(0, frame_length) множится по вертикали num_frames раз
    # Столбец смещений множится по горизонтали frame_length раз
    indices = (
        np.tile(np.arange(0, frame_length), (num_frames, 1)) + 
        np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
    )
    
    # 2. Вырезаем фреймы из дополненного сигнала по сгенерированной матрице индексов
    frames = pad_signal[indices.astype(np.int32, copy=False)]
    
    # 3. Накладываем оконную функцию Хэмминга на каждый фрейм
    # np.hamming(frame_length) создает одномерное окно, 
    # NumPy автоматически применит его к каждой строке матрицы frames
    frames *= np.hamming(frame_length)

    ###########################################################

    return frames

def power_spectrum(frames, NFFT=512):
    # Here you need to compute power spectum of framed signal with NFFT fft bins number

    """
    :param frames: framed signal
    :param NFFT: number of fft bins
    :return: pow_frames: framed signal power spectrum
    """

    mag_frames = np.absolute(np.fft.rfft(frames, NFFT))  # Magnitude of the FFT

    ###########################################################
    # Here is your code to compute pow_frames
    # Спектр мощности — это квадрат модуля спектра (АЧХ), деленый на количество точек БПФ (NFFT)
    pow_frames = (1.0 / NFFT) * (mag_frames ** 2)

    ###########################################################

    return pow_frames

def compute_fbank_filters(nfilt=40, sample_rate=16000, NFFT=512):
    # Here you need to compute fbank filters (FBs) for special case (sample_rate & NFFT)

    """
    :param nfilt: number of filters
    :param sample_rate: signal sampling rate
    :param NFFT: number of fft bins in power spectrum
    :return: fbank [nfilt x (NFFT/2+1)]
    """
    
    low_freq_mel = 0
    high_freq = sample_rate / 2

    ###########################################################
    # Here is your code to convert Convert Hz to Mel: 
    # high_freq -> high_freq_mel
    # задает верхний уровень mel-шкалы
    high_freq_mel = 2595 * np.log10(1 + high_freq / 700)
    ###########################################################

    mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilt + 2) # equally spaced in mel scale

    ###########################################################
    # Here is your code to convert Convert Mel to Hz: 
    # обратный перевод из mel_points -> hz_points
    hz_points = 700 * (10**(mel_points / 2595) - 1)
    ###########################################################

    bin = np.floor((NFFT + 1) * hz_points / sample_rate)

    fbank = np.zeros((nfilt, int(np.floor(NFFT / 2 + 1))))
    for m in range(1, nfilt + 1):
        f_m_minus = int(bin[m - 1]) # left
        f_m = int(bin[m])           # center
        f_m_plus = int(bin[m + 1])  # right

        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])

    return fbank

def compute_fbanks_features(pow_frames, fbank):
    # You need to compute fbank features using power spectrum frames and suitable fbanks filters
    
    """
    :param pow_frames: framed signal power spectrum, matrix [nframes x sample_rate*frame_size]
    :param fbank: matrix of the fbank filters [nfilt x (NFFT/2+1)] where NFFT: number of fft bins in power spectrum
    :return: filter_banks_features: log mel FB energies matrix [nframes x nfilt]
    """
    
    ###########################################################
    # Here is your code to compute filter_banks_features
    # Перемножаем спектр мощности каждого фрейма на АЧХ фильтров.
    # Матричное умножение автоматически просуммирует коэффициенты внутри полос.
    filter_banks_features = np.dot(pow_frames, fbank.T)
    ###########################################################

    filter_banks_features = np.where(filter_banks_features == 0, np.finfo(float).eps,
                                     filter_banks_features) # numerical stability
    filter_banks_features = np.log(filter_banks_features)

    return filter_banks_features

def compute_mfcc(filter_banks_features, num_ceps=20):
    # Here you need to compute MFCCs features using precomputed log mel FB energies matrix
    
    """
    :param filter_banks_features: log mel FB energies matrix [nframes x nfilt]
    :param num_ceps: number of cepstral components for MFCCs
    :return: mfcc: mel-frequency cepstral coefficients (MFCCs)
    """
    
    ###########################################################
    # Here is your code to compute mfcc features
    # 1. Применяем дискретное косинусное преобразование (DCT-II) вдоль оси фильтров (axis=1)
    # Параметр norm='ortho' делает преобразование ортогональным (нормирует энергию)
    raw_dct = dct(filter_banks_features, type=2, axis=1, norm='ortho')
    
    # 2. Берем первые num_ceps коэффициентов для каждого фрейма
    mfcc = raw_dct[:, :num_ceps]
    ###########################################################

    return mfcc

def mvn_floating(features, LC, RC, unbiased=False):
    # Here you need to do mean variance normalization of the input features
    
    """
    :param features: features matrix [nframes x nfeats]
    :param LC: the number of frames to the left defining the floating
    :param RC: the number of frames to the right defining the floating
    :param unbiased: biased or unbiased estimation of normalising sigma
    :return: normalised_features: normalised features matrix [nframes x nfeats]
    """
    
    nframes, dim = features.shape
    LC = min(LC, nframes - 1)
    RC = min(RC, nframes - 1)
    n = (np.r_[np.arange(RC + 1, nframes), np.ones(RC + 1) * nframes] - np.r_[np.zeros(LC), np.arange(nframes - LC)])[:,
        np.newaxis]
    f = np.cumsum(features, 0)
    s = np.cumsum(features ** 2, 0)
    f = (np.r_[f[RC:], np.repeat(f[[-1]], RC, axis=0)] - np.r_[np.zeros((LC + 1, dim)), f[:-LC - 1]]) / n
    s = (np.r_[s[RC:], np.repeat(s[[-1]], RC, axis=0)] - np.r_[np.zeros((LC + 1, dim)), s[:-LC - 1]]
         ) / (n - 1 if unbiased else n) - f ** 2 * (n / (n - 1) if unbiased else 1)
    
    ###########################################################
    # Here is your code to compute normalised features
    # 1. Защищаем дисперсию от отрицательных или нулевых значений
    # Если где-то дисперсия оказалась <= 0, временно заменяем на eps
    s_safe = np.where(s <= 0, np.finfo(float).eps, s)
    
    # 2. Вычисляем стандартное отклонение (корень из дисперсии)
    sigma = np.sqrt(s_safe)
    
    # 3. Вычитаем среднее (f) и делим на стандартное отклонение (sigma)
    normalised_features = (features - f) / sigma
    ###########################################################

    normalised_features[s == 0] = 0

    return normalised_features