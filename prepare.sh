mkdir data
cd data
wget http://qwone.com/~jason/20Newsgroups/20news-18828.tar.gz
tar -xvf 20news-18828.tar.gz
rm 20news-18828.tar.gz
cd ..
./download_fp.sh
python -c "import nltk; nltk.download('punkt')"
wandb login
