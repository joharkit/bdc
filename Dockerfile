FROM python:3.8.1-buster

RUN pip3 install matplotlib scipy astropy h5py

RUN pip3 install git+https://gitlab.mpcdf.mpg.de/ift/nifty_gridder.git@65e64dfd08953935e650ee8acbe753bcefad40eb

RUN pip3 install git+https://gitlab.mpcdf.mpg.de/ift/nifty@12f3469208e17958218d8b50dc80f33cad426e9d

RUN pip3 install git+https://gitlab.mpcdf.mpg.de/mtr/pypocketfft.git@2147aaf37c6274cc77160f6b355ed9d120cfdc57

ENV MPLBACKEND agg

WORKDIR /work
