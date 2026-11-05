# Exercises in order to perform laboratory work


# Import of modules
import numpy as np
import scipy.signal

from skimage.morphology import opening, closing


def load_vad_markup(path_to_rttm, signal, fs):
    # Функция для чтения rttm файлов и генерации разметки VAD в отсчетах (семплах)
    
    # Создаем массив из нулей (тишина) длиной как сам сигнал, тип данных - float32
    vad_markup = np.zeros(len(signal)).astype('float32')
        
    ###########################################################
    # Открываем RTTM-файл в режиме чтения ('r') с автоматическим закрытием
    with open(path_to_rttm, 'r') as file:
        # Построчно читаем весь файл в цикле
        for line in file:
            # Разбиваем строку по пробелам на отдельные элементы (слова/числа)
            parts = line.strip().split()
            
            # Проверяем, что строка не пустая и содержит нужные нам данные (минимум 5 колонок) - нам хватит 5 из 10
            if len(parts) >= 5:
                # Извлекаем время начала речевого сегмента (4-й элемент, переводим в секунды)
                start_sec = float(parts[3])
                # Извлекаем длительность речевого сегмента (5-й элемент, переводим в секунды)
                duration_sec = float(parts[4])
                
                # Вычисляем индекс начального отсчета (семпла) в массиве сигнала
                start_sample = int(start_sec * fs)
                # Вычисляем индекс конечного отсчета (семпла) в массиве сигнала
                end_sample = int((start_sec + duration_sec) * fs)
                
                # Ограничиваем конечный индекс длиной сигнала, чтобы случайно не выйти за границы массива
                end_sample = min(end_sample, len(signal))
                
                # Заполняем этот интервал единицами (обозначаем присутствие речи)
                vad_markup[start_sample:end_sample] = 1.0
    ###########################################################
    
    # Возвращаем готовый массив с разметкой детектора речевой активности
    return vad_markup

def framing(signal, window=320, shift=160):
    # Функция разбиения сигнала на перекрывающиеся сегменты (кадры / фреймы)
    # Вычисляем количество фреймов на основе длины сигнала, размера окна и шага
    num_frames = int((signal.shape[0] - window) / shift + 1)

    # Задаем кортеж с финальной формой двумерного массива (количество фреймов, размер окна)
    shape = (num_frames, window)
    # Выделяем память под массив фреймов нужного размера и типа данных float32
    frames = np.zeros(shape).astype('float32')

    ###########################################################
    # Заполняем массив сегментами исходного сигнала в цикле
    for idx in range(num_frames):
        # Вычисляем индекс начала текущего фрейма в отсчетах сигнала
        start = idx * shift
        # Извлекаем отрезок сигнала длиной в одно окно и сохраняем как строку матрицы
        frames[idx] = signal[start : start + window]
    
    ###########################################################
    
    return frames

def frame_energy(frames):
    # Function to compute frame energies
    # Функция для вычисления энергии каждого сегмента (кадра)

    # Инициализируем массив из нулей нужной длины (по количеству фреймов)
    E = np.zeros(frames.shape[0]).astype('float32')

    ###########################################################
    # Here is your code
    # Запускаем цикл по каждому фрейму (строке в матрице фреймов)
    for idx in range(frames.shape[0]):
        # Берем текущий фрейм (строку под номером idx)
        current_frame = frames[idx]
        
        # Находим энергию фрейма: суммируем все отсчеты внутри него.
        # (возводить в квадрат здесь не нужно, так как в функции 
        # energy_gmm_vad на вход уже поданы квадраты сигнала: squared_signal = signal**2)
        E[idx] = np.sum(current_frame)
    
    ###########################################################
    
    return E

def norm_energy(E):
    # Function to normalize energy by mean energy and energy standard deviation
    
    E_norm = np.zeros(len(E)).astype('float32')

    ###########################################################
    # Here is your code
    # Вычисляем среднее значение энергии по всем сегментам звукозаписи (m_E)
    m_E = np.mean(E)
    
    # Вычисляем среднеквадратическое отклонение энергии (sigma_E)
    # Используем ddof=1 для несмещенной оценки (деление на M - 1, как в формуле задания)
    sigma_E = np.std(E, ddof=1)
    
    # Заполняем созданный массив E_norm нормализованными значениями.
    # Добавляем маленькое число 1e-12 в знаменатель для защиты от деления на ноль.
    E_norm[:] = (E - m_E) / (sigma_E + 1e-12)
    
    ###########################################################
    
    return E_norm

def gmm_train(E, gauss_pdf, n_realignment):
    # Функция обучения параметров одномерной модели гауссовых смесей (GMM) с помощью EM-алгоритма
    
    # Инициализация априорных вероятностей (весов) трех компонент смеси (равные доли)
    w     = np.array([ 0.33, 0.33, 0.33])
    # Инициализация математических ожиданий (компонента 0 — шум, компоненты 1 и 2 — речь)
    m     = np.array([-1.00, 0.00, 1.00])
    # Инициализация среднеквадратических отклонений (сигм) компонент единичными значениями
    sigma = np.array([ 1.00, 1.00, 1.00])

    # Выделяем матрицу апостериорных вероятностей g размера [число_фреймов x число_компонент]
    g = np.zeros([len(E), len(w)])
    for n in range(n_realignment):

        # E-step
        ###########################################################
        # Here is your code
        for j in range(len(w)):
            # Вычисляем числитель формулы Байеса: априорный вес * плотность вероятности Гаусса
            g[:, j] = w[j] * gauss_pdf(E, m[j], sigma[j])
            
        # Находим знаменатель полной вероятности — сумму числителей по всем компонентам для каждого фрейма
        # Добавляем небольшую константу 1e-12 во избежание деления на ноль
        total_prob = np.sum(g, axis=1, keepdims=True) + 1e-12
        
        # Делим числители на полную вероятность, получая итоговые апостериорные вероятности g_ij
        g = g / total_prob

        ###########################################################

        # M-step
        ###########################################################
        # Here is your code
        # Вычисляем сумму апостериорных вероятностей для каждой компоненты
        sum_g = np.sum(g, axis=0) + 1e-12
        
        # 1. Обновляем априорные вероятности (веса) компонент смеси: w_j = (сумма g_ij) / M
        w = sum_g / len(E)
        
        # В циклах обновляем математические ожидания и сигмы для каждой j-й компоненты
        for j in range(len(w)):
            # 2. Обновляем матожидание: взвешенное среднее нормированных энергий фреймов
            m[j] = np.sum(g[:, j] * E) / sum_g[j]
            
            # 3. Обновляем дисперсию и берем корень для получения среднеквадратического отклонения sigma_j
            variance = np.sum(g[:, j] * ((E - m[j]) ** 2)) / sum_g[j]
            sigma[j] = np.sqrt(variance)

        ###########################################################
    
    return w, m, sigma

def eval_frame_post_prob(E, gauss_pdf, w, m, sigma):
    # Function to estimate a posterior probability that frame isn't speech
    # Функция для оценки апостериорной вероятности того, что кадр относится к «не речи» (шуму)

    g0 = np.zeros(len(E))

    ###########################################################
    # Here is your code
    # Создаем временный массив для хранения числителя формулы Байеса (для 0-й компоненты шума)
    # Формула: w_0 * N(E_norm; m_0, sigma_0)
    numerator = w[0] * gauss_pdf(E, m[0], sigma[0])
    
    # Переменная для накопления полной вероятности (знаменателя формулы Байеса)
    total_prob = np.zeros(len(E))
    
    # В цикле суммируем плотности вероятностей всех 3-х компонент, взвешенные по их априорным весам
    for s in range(len(w)):
        total_prob += w[s] * gauss_pdf(E, m[s], sigma[s])
        
    # Добавляем маленькую константу 1e-12 к знаменателю для предотвращения деления на ноль
    total_prob += 1e-12
    
    # Записываем результат деления числителя на знаменатель в уже созданный массив g0
    g0[:] = numerator / total_prob

    ###########################################################
            
    return g0

def energy_gmm_vad(signal, window, shift, gauss_pdf, n_realignment, vad_thr, mask_size_morph_filt):
    # Function to compute markup energy voice activity detector based of gaussian mixtures model
    
    # Squared signal
    squared_signal = signal**2
    
    # Frame signal with overlap
    frames = framing(squared_signal, window=window, shift=shift)
    
    # Sum frames to get energy
    E = frame_energy(frames)
    
    # Normalize the energy
    E_norm = norm_energy(E)
    
    # Train parameters of gaussian mixture models
    # Запуск EM-обучения весов, матожиданий и дисперсий трех гауссиан
    w, m, sigma = gmm_train(E_norm, gauss_pdf, n_realignment=10)
    
    # Estimate a posterior probability that frame isn't speech
    # Расчет вероятности отсутствия речи во фреймах
    g0 = eval_frame_post_prob(E_norm, gauss_pdf, w, m, sigma)
    
    # Compute real VAD's markup
    # Если вероятность шума g0 меньше заданного порога vad_thr, значит в кадре есть РЕЧЬ (ставим 1)
    vad_frame_markup_real = (g0 < vad_thr).astype('float32')  # frame VAD's markup
    
    # Разворачивание пофреймовой разметки обратно в разметку по отсчетам исходного сигнала
    vad_markup_real = np.zeros(len(signal)).astype('float32') # sample VAD's markup
    for idx in range(len(vad_frame_markup_real)):
        vad_markup_real[idx*shift:shift+idx*shift] = vad_frame_markup_real[idx]

    vad_markup_real[len(vad_frame_markup_real)*shift - len(signal):] = vad_frame_markup_real[-1]
    
    # Сглаживание бинарного решения с помощью морфологических фильтров (убираем короткие вспышки шума и паузы)
    vad_markup_real = closing(vad_markup_real, np.ones(mask_size_morph_filt)) # close filter
    vad_markup_real = opening(vad_markup_real, np.ones(mask_size_morph_filt)) # open filter
    
    return vad_markup_real

def reverb(signal, impulse_response):
    # Function to create reverberation effect
    
    signal_reverb = np.zeros(len(signal)).astype('float32')
    
    ###########################################################
    # Here is your code
    # Берем два массива (голос и готовую характеристику эха из файла)
    # и перемножаем их со сдвигом с помощью стандартной функции свертки из scipy.
    # Флаг mode='full' означает, что мы считаем полную длину получившихся «хвостов».
    # Длина массива full_convolution будет равна: len(signal) + len(impulse_response) - 1.
    full_convolution = scipy.signal.convolve(signal, impulse_response, mode='full')
    
    # чтобы результат не стал длиннее, чем оригинал, считаем отсчеты
    original_length = len(signal)
    
    # Отрезаем от длинной свертки кусок, равный длине оригинального сигнала.
    # Берем отсчеты от 0 до original_length.
    trimmed_result = full_convolution[0 : original_length]
    
    # Копируем эти значения внутрь заготовки signal_reverb
    # Синтаксис [:] означает "заменить цифры внутри созданного массива на новые".
    signal_reverb[:] = trimmed_result
    ###########################################################
    
    return signal_reverb

def awgn(signal, sigma_noise):
    # Function to add white gaussian noise to signal
    
    signal_noise = np.zeros(len(signal)).astype('float32')
    
    ###########################################################
    # Here is your code
    # Шаг 1: Генерируем массив случайных чисел (шум) такой же длины, как сигнал.
    # Распределение Гаусса (нормальное): центр (loc) = 0, шаг разброса (scale) = sigma_noise.
    white_noise = np.random.normal(loc=0.0, scale=sigma_noise, size=len(signal))
    
    # Шаг 2: Накладываем (плюсуем) сгенерированный шум на чистый сигнал.
    # Используем срез [:], чтобы записать данные прямо в массив, созданный ранее.
    signal_noise[:] = signal + white_noise
    
    ###########################################################
    
    return signal_noise