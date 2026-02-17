import numpy as np
import scipy

class LightCurve:
    def __init__(self, times:np.ndarray, data: np.ndarray, dt: float):
        self.data = np.asarray(data)
        self.times = np.asarray(times)
        self.dt = dt
        self.n = len(self.data)
        self.fs = 1 / dt  # sampling frequency

    @classmethod
    def from_file(cls, filename, delimiter="\t"):
        times, data = np.loadtxt(filename, delimiter=delimiter, dtype=None, unpack=True,usecols =(0, 1))
        dt = float(times[1] - times[0])
        return cls(times, data, dt)

    @property
    def time(self):
        return np.arange(self.n) * self.dt

    def segment(self, seglength):
        ####Segment time series
        if seglength > self.n:
            seglength = self.n
            num_segments = 1
        else:
            num_segments = int(self.n/seglength)

        segnum=np.arange(0,num_segments)
        segments = []
        for s in segnum:
                start = s*seglength
                end = start+seglength
                segments.append(self.data[start:end] * self.dt)  #convert from cts/s to cts/bin 

        return np.array(segments)



class BispectralAnalysis:
    def __init__(self, lc: LightCurve):
        self.lc = lc
        self.data = lc.data
        self.dt = lc.dt
        self.fs = lc.fs


    def power_spectrum(self, seglength, normalization="leahy"):
        dt = self.dt

        segments = self.lc.segment(seglength)
        num_segments, n = segments.shape

        power_spec = np.zeros([seglength])

        ####Calculate Fourier Frequencies
        seg_freq = np.fft.fftfreq(seglength, d=self.dt)
        positive = seg_freq >= 0
    
        for s in segments:

            ####FFT of segment
            seg_ps = scipy.fftpack.fft(s)
            N_ph = sum(s) #number of photons
            t_seg = seglength * dt #length of segment in seconds
            mean_rate = N_ph/t_seg #mean count rate of photons
            power_spec_current = np.absolute(seg_ps * np.conj(seg_ps))

            ####Normalise, Leahy
            if normalization == 'leahy':
                power_spec_current = power_spec_current*2.0/N_ph

            ####Normalize, Fractional rms
            elif normalization == 'frac_rms':
            #Assumes white noise level of 2, but value might be lower due to deadtime effects!    
                power_spec_current = ((power_spec_current*2.0/N_ph - 2.0) / mean_rate)

            ####Normalize, absolute rms
            elif normalization == 'abs_rms':
                power_spec_current = power_spec_current*2.0*mean_rate/N_ph

            else:
                raise ValueError("normalization must be 'leahy', 'frac_rms', or 'abs_rms'")

            power_spec += power_spec_current  
        
        ####Average power spectra
        power_spec = (power_spec)/num_segments

        ####Get error on power
        power_err = power_spec/np.sqrt(num_segments)

        ####Return results
        freqs = seg_freq[positive]
        power = power_spec[positive]
        pow_err = power_err[positive]

        return freqs, power, pow_err


    def bispectrum(self, seglength):
#        n = self.n
        dt = self.dt

        segments = self.lc.segment(seglength)
        num_segments, n = segments.shape

        if seglength%2 == 0:
            nfft = seglength//4 #generally set to seglength//4 for computational time and plotting reasons. Can be set to seglength//2 if needed for high QPO freq. 
        else:
            nfft = seglength//4 +1


        bspec=np.zeros([nfft,nfft],dtype=np.complex_)
        bcoha=np.zeros([nfft,nfft],dtype=np.complex_)
        bcohb=np.zeros([nfft,nfft],dtype=np.complex_)
        bicoher=np.zeros([nfft,nfft])
        err_bicoher = np.zeros([nfft,nfft])
        logbicoher=np.zeros([nfft,nfft])
        biphase = np.zeros([nfft,nfft])
        err_biphase = np.zeros([nfft,nfft])

        freqs=[]
        for s_index, s_value in enumerate(segments):
            #print('starting segment',s+1,'/',num_segments)
            print(f"Starting segment: {s_index + 1} / ", num_segments)

            ####FFT of segment
            seg_sp = np.fft.fft(s_value)
            seg_freq = np.fft.fftfreq(seglength, d=dt)
            if s_index==0:
                for i in range(nfft):
                    freqs.append(seg_freq[i])
            ####Calculate bispectrum
            for r in range(nfft):
                for c in range(nfft-r):
                    t1 = seg_sp[r]
                    t2 = seg_sp[c]
                    t3 = np.conjugate(seg_sp[r+c])
                    t4 = seg_sp[r+c]
                    bspec[r][c]+= t1*t2*t3
                    bcoha[r][c] += (abs(t1*t2))**2.0
                    bcohb[r][c] += (abs(t4))**2.0

        ####calculate bicoherence, b^2
        for r in range(nfft):
            for c in range(nfft-r):
                if bspec[r][c] == 0:
                    bicoher[r][c] = 0
                else:
                    bicoher[r][c] = (abs(bspec[r][c]))**2.0 / (bcoha[r][c]*bcohb[r][c])

        ####Normalise bispectrum
        if num_segments > 1:
            bspec = bspec/num_segments
        else:
            bspec = bspec

        #### subtract bias

        bias = 1.0/num_segments
        bicoher = bicoher - bias
        bicoher[bicoher < 0] = 0

        #### calculate std deviation of bicoherence sqrt(var[b^2])

        err_bicoher = np.sqrt((2.0*bicoher/num_segments) * ((1.0 - bicoher)**3.0))


        ####Calculate log of bicoherence
        logbicoher = np.log10(bicoher)

        ####Calculate biphase
        phi = np.angle(bspec)
        biphase = phi


        #### Returning only values of interest
        cutoff = len(bicoher)//2
        freqs = freqs[1:cutoff]
        bicoher = bicoher[1:cutoff, 1:cutoff]
        logbicoher = logbicoher[1:cutoff, 1:cutoff]
        biphase = biphase[1:cutoff, 1:cutoff]

        ####return results
        return freqs,bicoher, err_bicoher, logbicoher, biphase

