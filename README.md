[![DOI](https://zenodo.org/badge/686800171.svg)](https://zenodo.org/badge/latestdoi/686800171)

#Reference

Supplementary material for the paper "A Survey on Gravity Waves in the Brazilian Sector Based on Radiosonde Measurements from 32 Aerodromes".

# Writting

The ´read_data_fft3_GH.py´ is executed with Python 3.6 and was created by A. Brhian and M. Ridenti.

In the CMD you need write:

```
CD C:\<YOUR DIRECTORY>\python read_data_fft3_GH.py data_test input_samples
```
# The inputs

Where `data_test` is file with spreadsheets in extension .csv, separeted by " " (on space) and with "." as decimal. Theses arquives have the following columns:

if temperature data (\*_temp.csv):
- NSINOTICO
- SIGLA (acronym)
- ANO (year)
- MES (month)
- DIA (day)
- HORA (hour)
- MINUTO (minute)
- MINSONDAGEM (minute of sounding)
- SEGSONDAGEM (second of sounding)
- TEMP (temperature in °C)
- UR (relativity humidy %)
- PO (dew point)
- ALTITUDE 
- PRESSAO (pression)

if wind data (\*_vento.csv):
- NSINOTICO
- SIGLA
- ANO
- MES
- DIA
- HORA
- MINUTO
- MINSONDAGEM
- SEGSONDAGEM
- DIRVENTO (direction from wind)
- VELVENTO (intensity of wind)
- ALTITUDE
- PRESSAO

and \* is the acronym from aerodrome.

The ´input_sample´ is an file with text archives which have following arguments:

- YEAR (YYYY)
- Height minimum
- Height maximum
- Height limit
- Height minimum for Fast Fourier Transforming (FFT)
- Height maximum for FFT
- If troposphere (T) or lower stratosphere (S)

# The outputs

In the file ´analytic´ is created the spreadsheets ´bd_Brunt´, ´bd_Ene´, ´bd_Est´ and, ´bd_FFT´. Once a time created, the next aerodromes are added or updated.

The ´bd_Brunt´ is saved the following infos:
- aerodromo
- ano
- camada: If troposphere (T) or lower stratosphere (S)
- estacao: If ´al´ days of year or other season
- variavel_N: The mean (N_mean) or standard deviation (N_sd) of Brunt-Väisälä frequency
- data_compilacao: Date of compilation of data in ´read_data_fft3_GH.py´
- hora_compilacao: Hour of compilation of data in ´read_data_fft3_GH.py´
- multDia: Quantity of measurement by day
- 0.0... 367: DOY stated by 0 on january, 1st

The ´bd_Ene´ is saved the following infos:
- aerodromo
- ano
- camada
- estacao
- tipo: If kinetic (K), potencial (P) or mechanical (M) energy
- data_compilacao
- hora_compilacao
- num_dados: Quantity of measurement
- erros_data: When hapened error in FFT calculate
- MK_slope: Tendency test of Mann-Kendall
- MK_p: p-value of Mann-Kendal test
- lin_slope: Slope from linear function between energy and days
- lin_intercept: Intercept from linear function between energy and days
- 0.0 ... 367

The ´bd_Est´ is saved the following infos:
- aerodromo
- ano
- camada
- estacao
- data_compilacao
- hora_compilacao
- R_pearson: R from Pearson's Correlation between kinetic and potencial energy
- p-value: Teste for Pearson's Correlation null
- slope: Slope from linear function between kinetic and potencial energy
- intercept: Intercept from linear function between kinetic and potencial energy
- n: Quantity of pair for correlation between energies
- std_err: Standard deviance from Pearson's correlation

The ´bd_FFT´ is saved the following infos:
- aerodromo
- ano
- camada
- estacao
- variavel_FFT: _amp (for amplitude profile calculated by FFT), _lamb (for wavelenght calculated by FFT), _par (coeficiente from parabolic ajust in profile), _pha (for phase calculated by FFT), _rel (the quality of ajust of parabolic and harmonic function). Theses variables can staterd with ew_ (eastward wind), nw_ (northward wind) and, te_ (temperature)
- data_compilacao
- hora_compilacao
- multDia: for multiplicity of measurements
- 0.0 ... 367

Some simples figures are saved in ´analytic´ just only for helping in the visualization of data.


