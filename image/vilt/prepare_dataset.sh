WORKSPACE=$(pwd)

RAW_ROOT=$1

ARROW_ROOT=$RAW_ROOT/arrow

if [ -z $RAW_ROOT ]
then
    echo "Usage: $0 <DATA_ROOT>"
    exit 1
fi

if [ ! -e $ARROW_ROOT ]
then
    mkdir $ARROW_ROOT
fi

if [ -e $RAW_ROOT ]
then
    cd $RAW_ROOT
else
    mkdir $RAW_ROOT
fi




# download all files
wget http://images.cocodataset.org/zips/train2014.zip
wget http://images.cocodataset.org/zips/val2014.zip
wget https://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip

# unzip all files
unzip train2014.zip -d $RAW_ROOT/train2014
unzip val2014.zip -d $RAW_ROOT/val2014
unzip caption_datasets.zip -d $RAW_ROOT/karpathy

# remove all files
rm train2014.zip
rm val2014.zip
rm caption_datasets.zip

# converting the dataset
cd $WORKSPACE
python utils/makearrow.py $RAW_ROOT $ARROW_ROOT
