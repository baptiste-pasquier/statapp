cd "work/"
git clone https://github.com/baptiste-pasquier/statapp/

wget --user-agent Mozilla/4.0 'https://transfer.sh/get/10rB5r/data.zip'
unzip data.zip
mv -v "data" "statapp/data"
rm data.zip

cd "statapp/"
conda env create
source activate statapp
python -m ipykernel install --user --name=statappenv
