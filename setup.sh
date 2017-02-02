mkdir active_labeling_database
cd ~
wget https://repo.continuum.io/archive/Anaconda2-4.3.0-Linux-x86_64.sh
bash Anaconda2-4.3.0-Linux-x86_64.sh -b -p $HOME/anaconda2
echo PATH="$HOME/anaconda2/bin:\$PATH" >> ~/.bashrc
source ~/.bashrc
conda install -c conda-forge typing=3.5.3.0 -y
conda install -c spyking-circus progressbar2=3.6.1 -y
pip install tensorflow

echo "Setup finished. Don't forget to manually edit checkpoints/1485596219/checkpoint to change the path."