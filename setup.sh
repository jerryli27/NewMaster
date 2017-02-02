cd ~
wget https://repo.continuum.io/archive/Anaconda2-4.3.0-Linux-x86_64.sh
bash Anaconda2-4.3.0-Linux-x86_64.sh -b -p $HOME/anaconda2
echo export PATH="$HOME/miniconda/bin:$PATH" >> ~/.bashrc
source ~/.bashrc
conda install -c conda-forge typing=3.5.3.0 -y
conda install -c spyking-circus progressbar2=3.6.1 -y

