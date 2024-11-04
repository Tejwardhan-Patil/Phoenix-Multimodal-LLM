#include <iostream>
#include <vector>
#include <fftw3.h>  
#include <sndfile.h> 
#include <cmath>
#include <complex>

// Constants
constexpr int NUM_MFCC = 13;  // Number of MFCC coefficients to extract
constexpr int FFT_SIZE = 1024;  // Size of FFT window
constexpr int HOP_SIZE = 512;  // Hop size for overlapping windows

// Hamming Window function
std::vector<double> generate_hamming_window(int size) {
    std::vector<double> window(size);
    for (int i = 0; i < size; ++i) {
        window[i] = 0.54 - 0.46 * std::cos(2 * M_PI * i / (size - 1));
    }
    return window;
}

// Function to compute the FFT of the audio signal
std::vector<std::complex<double>> compute_fft(const std::vector<double>& signal) {
    int N = signal.size();
    fftw_complex *in, *out;
    fftw_plan p;
    
    in = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N);
    out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N);
    
    for (int i = 0; i < N; ++i) {
        in[i][0] = signal[i];
        in[i][1] = 0.0;
    }

    p = fftw_plan_dft_1d(N, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_execute(p);

    std::vector<std::complex<double>> result(N);
    for (int i = 0; i < N; ++i) {
        result[i] = std::complex<double>(out[i][0], out[i][1]);
    }

    fftw_destroy_plan(p);
    fftw_free(in);
    fftw_free(out);

    return result;
}

// Function to convert frequency bin to Mel scale
double freq_to_mel(double freq) {
    return 2595.0 * std::log10(1.0 + freq / 700.0);
}

// Function to convert Mel scale back to frequency
double mel_to_freq(double mel) {
    return 700.0 * (std::pow(10.0, mel / 2595.0) - 1.0);
}

// Function to generate mel filter banks
std::vector<std::vector<double>> generate_mel_filter_banks(int num_filters, int fft_size, int sample_rate) {
    std::vector<std::vector<double>> filter_banks(num_filters, std::vector<double>(fft_size / 2 + 1, 0.0));

    double mel_min = freq_to_mel(0);
    double mel_max = freq_to_mel(sample_rate / 2);

    std::vector<double> mel_points(num_filters + 2);
    for (int i = 0; i < num_filters + 2; ++i) {
        mel_points[i] = mel_min + (mel_max - mel_min) * i / (num_filters + 1);
    }

    std::vector<int> bin_points(num_filters + 2);
    for (int i = 0; i < num_filters + 2; ++i) {
        bin_points[i] = std::floor((fft_size + 1) * mel_to_freq(mel_points[i]) / sample_rate);
    }

    for (int i = 1; i <= num_filters; ++i) {
        for (int j = bin_points[i - 1]; j < bin_points[i]; ++j) {
            filter_banks[i - 1][j] = (j - bin_points[i - 1]) / static_cast<double>(bin_points[i] - bin_points[i - 1]);
        }
        for (int j = bin_points[i]; j < bin_points[i + 1]; ++j) {
            filter_banks[i - 1][j] = (bin_points[i + 1] - j) / static_cast<double>(bin_points[i + 1] - bin_points[i]);
        }
    }

    return filter_banks;
}

// Function to apply the filter banks to the magnitude spectrum
std::vector<double> apply_mel_filter_banks(const std::vector<double>& magnitude_spectrum, const std::vector<std::vector<double>>& mel_filter_banks) {
    std::vector<double> mel_energies(mel_filter_banks.size(), 0.0);
    for (size_t i = 0; i < mel_filter_banks.size(); ++i) {
        for (size_t j = 0; j < magnitude_spectrum.size(); ++j) {
            mel_energies[i] += magnitude_spectrum[j] * mel_filter_banks[i][j];
        }
    }
    return mel_energies;
}

// Function to perform log scaling on mel energies
std::vector<double> log_scale(const std::vector<double>& mel_energies) {
    std::vector<double> log_mel_energies(mel_energies.size());
    for (size_t i = 0; i < mel_energies.size(); ++i) {
        log_mel_energies[i] = std::log(mel_energies[i] + 1e-6);
    }
    return log_mel_energies;
}

// Function to compute the Discrete Cosine Transform (DCT)
std::vector<double> compute_dct(const std::vector<double>& input, int num_coeffs) {
    std::vector<double> dct_output(num_coeffs, 0.0);
    for (int i = 0; i < num_coeffs; ++i) {
        for (size_t j = 0; j < input.size(); ++j) {
            dct_output[i] += input[j] * std::cos(M_PI * i * (j + 0.5) / input.size());
        }
    }
    return dct_output;
}

// Function to extract MFCC features from an audio signal
std::vector<std::vector<double>> extract_mfcc(const std::vector<double>& audio_data, int sample_rate, int num_mfcc = NUM_MFCC) {
    int num_frames = (audio_data.size() - FFT_SIZE) / HOP_SIZE + 1;
    std::vector<std::vector<double>> mfccs(num_frames, std::vector<double>(num_mfcc));

    std::vector<double> hamming_window = generate_hamming_window(FFT_SIZE);
    std::vector<std::vector<double>> mel_filter_banks = generate_mel_filter_banks(26, FFT_SIZE, sample_rate);

    for (int frame = 0; frame < num_frames; ++frame) {
        std::vector<double> frame_data(audio_data.begin() + frame * HOP_SIZE, audio_data.begin() + frame * HOP_SIZE + FFT_SIZE);

        // Apply Hamming window
        for (int i = 0; i < FFT_SIZE; ++i) {
            frame_data[i] *= hamming_window[i];
        }

        // Compute FFT and magnitude spectrum
        std::vector<std::complex<double>> fft_result = compute_fft(frame_data);
        std::vector<double> magnitude_spectrum(FFT_SIZE / 2 + 1);
        for (int i = 0; i < FFT_SIZE / 2 + 1; ++i) {
            magnitude_spectrum[i] = std::abs(fft_result[i]);
        }

        // Apply mel filter banks
        std::vector<double> mel_energies = apply_mel_filter_banks(magnitude_spectrum, mel_filter_banks);

        // Log scaling
        std::vector<double> log_mel_energies = log_scale(mel_energies);

        // Compute DCT to obtain MFCCs
        mfccs[frame] = compute_dct(log_mel_energies, num_mfcc);
    }

    return mfccs;
}

// Function to extract Chroma features from an audio signal
std::vector<std::vector<double>> extract_chroma(const std::vector<double>& audio_data, int sample_rate) {
    int num_frames = (audio_data.size() - FFT_SIZE) / HOP_SIZE + 1;
    std::vector<std::vector<double>> chroma_features(num_frames, std::vector<double>(12));

    std::vector<double> hamming_window = generate_hamming_window(FFT_SIZE);

    for (int frame = 0; frame < num_frames; ++frame) {
        std::vector<double> frame_data(audio_data.begin() + frame * HOP_SIZE, audio_data.begin() + frame * HOP_SIZE + FFT_SIZE);

        // Apply Hamming window
        for (int i = 0; i < FFT_SIZE; ++i) {
            frame_data[i] *= hamming_window[i];
        }

        // Compute FFT and magnitude spectrum
        std::vector<std::complex<double>> fft_result = compute_fft(frame_data);
        std::vector<double> magnitude_spectrum(FFT_SIZE / 2 + 1);
        for (int i = 0; i < FFT_SIZE / 2 + 1; ++i) {
            magnitude_spectrum[i] = std::abs(fft_result[i]);
        }

        // Compute chroma feature
        for (int i = 0; i < 12; ++i) {
            chroma_features[frame][i] = 0.0;
        }

        for (int bin = 0; bin < FFT_SIZE / 2 + 1; ++bin) {
            double freq = bin * sample_rate / FFT_SIZE;
            int chroma_bin = static_cast<int>(12 * std::log2(freq / 440.0)) % 12;
            chroma_features[frame][chroma_bin] += magnitude_spectrum[bin];
        }

        // Normalize chroma vector
        double chroma_sum = 0.0;
        for (int i = 0; i < 12; ++i) {
            chroma_sum += chroma_features[frame][i];
        }
        for (int i = 0; i < 12; ++i) {
            chroma_features[frame][i] /= (chroma_sum + 1e-6);
        }
    }

    return chroma_features;
}

// Function to read an audio file
std::vector<double> read_audio_file(const std::string& file_path, int& sample_rate) {
    SF_INFO sfinfo;
    SNDFILE *infile = sf_open(file_path.c_str(), SFM_READ, &sfinfo);

    if (!infile) {
        std::cerr << "Error reading audio file: " << file_path << std::endl;
        return {};
    }

    sample_rate = sfinfo.samplerate;
    std::vector<double> audio_data(sfinfo.frames);

    sf_read_double(infile, audio_data.data(), sfinfo.frames);
    sf_close(infile);

    return audio_data;
}

// Function to normalize audio data
std::vector<double> normalize_audio(const std::vector<double>& audio_data) {
    std::vector<double> normalized_data = audio_data;

    double max_amplitude = 0.0;
    for (const auto& sample : audio_data) {
        if (std::abs(sample) > max_amplitude) {
            max_amplitude = std::abs(sample);
        }
    }

    if (max_amplitude > 0.0) {
        for (auto& sample : normalized_data) {
            sample /= max_amplitude;
        }
    }

    return normalized_data;
}

// Function to split stereo audio into mono
std::vector<double> convert_to_mono(const std::vector<double>& audio_data, int channels) {
    if (channels == 1) {
        return audio_data;  // Already mono
    }

    std::vector<double> mono_data(audio_data.size() / channels);
    for (size_t i = 0; i < mono_data.size(); ++i) {
        double sum = 0.0;
        for (int ch = 0; ch < channels; ++ch) {
            sum += audio_data[i * channels + ch];
        }
        mono_data[i] = sum / channels;
    }

    return mono_data;
}

// Utility function to print feature matrix (for debugging purposes)
void print_feature_matrix(const std::vector<std::vector<double>>& features) {
    for (const auto& frame : features) {
        for (const auto& coeff : frame) {
            std::cout << coeff << " ";
        }
        std::cout << std::endl;
    }
}

// Main function
int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <audio_file_path>" << std::endl;
        return -1;
    }

    std::string audio_file = argv[1];
    int sample_rate;
    std::vector<double> audio_data = read_audio_file(audio_file, sample_rate);

    if (audio_data.empty()) {
        std::cerr << "Error: Unable to process audio file." << std::endl;
        return -1;
    }

    // Normalize audio data
    std::vector<double> normalized_audio = normalize_audio(audio_data);

    // Extract MFCC features
    std::vector<std::vector<double>> mfccs = extract_mfcc(normalized_audio, sample_rate);

    // Print MFCC features (for debugging)
    std::cout << "MFCC Features:" << std::endl;
    print_feature_matrix(mfccs);

    // Extract Chroma features
    std::vector<std::vector<double>> chroma = extract_chroma(normalized_audio, sample_rate);

    // Print Chroma features (for debugging)
    std::cout << "Chroma Features:" << std::endl;
    print_feature_matrix(chroma);

    return 0;
}

// Function to apply pre-emphasis to the audio signal
std::vector<double> apply_pre_emphasis(const std::vector<double>& audio_data, double pre_emphasis_coeff = 0.97) {
    std::vector<double> emphasized_audio(audio_data.size());
    emphasized_audio[0] = audio_data[0];
    for (size_t i = 1; i < audio_data.size(); ++i) {
        emphasized_audio[i] = audio_data[i] - pre_emphasis_coeff * audio_data[i - 1];
    }
    return emphasized_audio;
}

// Function to frame audio data into overlapping windows
std::vector<std::vector<double>> frame_audio(const std::vector<double>& audio_data, int frame_size, int hop_size) {
    int num_frames = (audio_data.size() - frame_size) / hop_size + 1;
    std::vector<std::vector<double>> frames(num_frames, std::vector<double>(frame_size));

    for (int i = 0; i < num_frames; ++i) {
        for (int j = 0; j < frame_size; ++j) {
            frames[i][j] = audio_data[i * hop_size + j];
        }
    }
    return frames;
}

// Function to extract RMS energy from audio frames
std::vector<double> compute_rms_energy(const std::vector<std::vector<double>>& frames) {
    std::vector<double> rms_energies(frames.size());
    for (size_t i = 0; i < frames.size(); ++i) {
        double sum_of_squares = 0.0;
        for (double sample : frames[i]) {
            sum_of_squares += sample * sample;
        }
        rms_energies[i] = std::sqrt(sum_of_squares / frames[i].size());
    }
    return rms_energies;
}

// Function to compute zero-crossing rate from audio frames
std::vector<double> compute_zero_crossing_rate(const std::vector<std::vector<double>>& frames) {
    std::vector<double> zcr(frames.size());
    for (size_t i = 0; i < frames.size(); ++i) {
        int zero_crossings = 0;
        for (size_t j = 1; j < frames[i].size(); ++j) {
            if ((frames[i][j - 1] > 0 && frames[i][j] <= 0) || (frames[i][j - 1] < 0 && frames[i][j] >= 0)) {
                zero_crossings++;
            }
        }
        zcr[i] = zero_crossings / static_cast<double>(frames[i].size());
    }
    return zcr;
}

// Function to extract spectral centroid from audio frames
std::vector<double> compute_spectral_centroid(const std::vector<std::vector<double>>& frames, int sample_rate) {
    std::vector<double> spectral_centroids(frames.size());

    for (size_t i = 0; i < frames.size(); ++i) {
        std::vector<std::complex<double>> fft_result = compute_fft(frames[i]);
        std::vector<double> magnitude_spectrum(fft_result.size());
        for (size_t j = 0; j < fft_result.size(); ++j) {
            magnitude_spectrum[j] = std::abs(fft_result[j]);
        }

        double weighted_sum = 0.0;
        double total_sum = 0.0;

        for (size_t j = 0; j < magnitude_spectrum.size(); ++j) {
            double frequency = j * sample_rate / magnitude_spectrum.size();
            weighted_sum += frequency * magnitude_spectrum[j];
            total_sum += magnitude_spectrum[j];
        }

        spectral_centroids[i] = (total_sum > 0) ? weighted_sum / total_sum : 0.0;
    }

    return spectral_centroids;
}

// Function to apply all feature extraction techniques
void extract_audio_features(const std::string& audio_file) {
    int sample_rate;
    std::vector<double> audio_data = read_audio_file(audio_file, sample_rate);

    if (audio_data.empty()) {
        std::cerr << "Error: Unable to process audio file." << std::endl;
        return;
    }

    // Pre-emphasis
    std::vector<double> emphasized_audio = apply_pre_emphasis(audio_data);

    // Normalize audio
    std::vector<double> normalized_audio = normalize_audio(emphasized_audio);

    // Convert to mono
    int channels = 1;
    std::vector<double> mono_audio = convert_to_mono(normalized_audio, channels);

    // Frame audio
    std::vector<std::vector<double>> frames = frame_audio(mono_audio, FFT_SIZE, HOP_SIZE);

    // Extract RMS Energy
    std::vector<double> rms_energies = compute_rms_energy(frames);
    std::cout << "RMS Energy per Frame:" << std::endl;
    for (double energy : rms_energies) {
        std::cout << energy << " ";
    }
    std::cout << std::endl;

    // Extract Zero Crossing Rate
    std::vector<double> zcr = compute_zero_crossing_rate(frames);
    std::cout << "Zero Crossing Rate per Frame:" << std::endl;
    for (double rate : zcr) {
        std::cout << rate << " ";
    }
    std::cout << std::endl;

    // Extract Spectral Centroid
    std::vector<double> spectral_centroids = compute_spectral_centroid(frames, sample_rate);
    std::cout << "Spectral Centroid per Frame:" << std::endl;
    for (double centroid : spectral_centroids) {
        std::cout << centroid << " ";
    }
    std::cout << std::endl;

    // Extract MFCC Features
    std::vector<std::vector<double>> mfccs = extract_mfcc(mono_audio, sample_rate);
    std::cout << "MFCC Features:" << std::endl;
    print_feature_matrix(mfccs);

    // Extract Chroma Features
    std::vector<std::vector<double>> chroma = extract_chroma(mono_audio, sample_rate);
    std::cout << "Chroma Features:" << std::endl;
    print_feature_matrix(chroma);
}

// Enhanced main function to handle multiple feature extraction
int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <audio_file_path>" << std::endl;
        return -1;
    }

    std::string audio_file = argv[1];
    extract_audio_features(audio_file);

    return 0;
}