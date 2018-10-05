set -e
set -u

apt-get install git htop
echo -n "Installing python3.6 .. "
echo 'deb http://ftp.de.debian.org/debian testing main' | sudo tee --append /etc/apt/sources.list
echo 'APT::Default-Release "stable";' | sudo tee -a /etc/apt/apt.conf.d/00local
apt-get update
apt-get -t testing -y --force-yes install python3.6 python3-venv

python3.6 -m venv venv
source ./venv/bin/activate
pip install pandas sklearn

mkdir ~/.kaggle
mv kaggle.json ~/.kaggle
kaggle competitions download -c avazu-ctr-prediction

kaggle competitions submit -c avazu-ctr-prediction -f submission.csv -m "Message"
