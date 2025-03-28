use std::io;
use std::io::{Error, ErrorKind};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::fs::{create_dir_all, File};

use byteorder::{BigEndian, ReadBytesExt};
use flate2::read::GzDecoder;

const LABEL_MAGIC_NO: u32 = 2049;
const IMG_MAGIC_NO: u32 = 2051;

pub struct FashionMNIST {
    pub train_labels: Vec<u8>,
    pub train_imgs: Vec<Vec<u8>>,
    pub test_labels: Vec<u8>,
    pub test_imgs: Vec<Vec<u8>>
}

pub struct FashionMNISTBuilder {
    data_home: String,
    force_download: bool,
    verbose: bool
}

impl FashionMNISTBuilder {
    pub fn new() -> FashionMNISTBuilder {
        FashionMNISTBuilder {
            data_home: "fashion_mnist".into(),
            force_download: false,
            verbose: false
        }
    }

    pub fn data_home<S: Into<String>>(mut self, dh: S) -> FashionMNISTBuilder {
        self.data_home = dh.into();
        self
    }

    pub fn verbose(mut self) -> FashionMNISTBuilder {
        self.verbose = true;
        self
    }

    pub fn get_data(self) -> io::Result<FashionMNIST> {
        if self.verbose {
            println!("Creating data directory: {}", self.data_home);
        }
        if self.verbose { println!("Extracting data"); }
        let (_train_lbl_meta, train_labels) = self.extract_labels(
            self.get_file_path("train_gz/train-labels-idx1-ubyte.gz"))?;
        let (_train_img_meta, train_imgs) = self.extract_images(
            self.get_file_path("train_gz/train-images-idx3-ubyte.gz"))?;
        let (_test_lbl_meta, test_labels) = self.extract_labels(
            self.get_file_path("test_gz/t10k-labels-idx1-ubyte.gz"))?;
        let (_test_img_meta, test_imgs) = self.extract_images(
            self.get_file_path("test_gz/t10k-images-idx3-ubyte.gz"))?;

        println!("metadata:[Test] images={:?}, labels={:?}; [train] images={:?}, labels={:?}",
            _test_img_meta, _test_lbl_meta, _train_img_meta, _train_lbl_meta
        );
        if self.verbose { println!("FashionMNIST Loaded!"); }
        Ok(FashionMNIST {
            train_imgs: train_imgs,
            train_labels: train_labels,
            test_imgs: test_imgs,
            test_labels: test_labels
        })
    }

    fn get_file_path(&self, filename: &str) -> PathBuf {
        Path::new(&self.data_home).join(filename)
    }

    fn extract_labels<P: AsRef<Path>>(&self, label_file_path: P)
                                      -> io::Result<([u32; 2], Vec<u8>)>
    {
        let mut decoder = self.get_decoder(label_file_path)?;
        let mut metadata_buf = [0u32; 2];

        decoder.read_u32_into::<BigEndian>(&mut metadata_buf)?;

        let mut labels = Vec::new();
        decoder.read_to_end(&mut labels)?;
        if metadata_buf[0] != LABEL_MAGIC_NO {
            Err(Error::new(ErrorKind::InvalidData,
                           "Unable to verify FashionMNIST data. Force redownload."))
        } else {
            Ok((metadata_buf, labels))
        }
    }

    fn extract_images<P: AsRef<Path>>(&self, img_file_path: P)
                                      -> io::Result<([u32; 4], Vec<Vec<u8>>)>
    {
        let mut decoder = self.get_decoder(img_file_path)?;
        let mut metadata_buf = [0u32; 4];

        decoder.read_u32_into::<BigEndian>(&mut metadata_buf)?;

        let mut imgs = Vec::new();
        decoder.read_to_end(&mut imgs)?;
        if metadata_buf[0] != IMG_MAGIC_NO {
            Err(Error::new(ErrorKind::InvalidData,
                           "Unable to verify FashionMNIST data. Force redownload."))
        } else {
            Ok((metadata_buf, imgs.chunks(784).map(|x| x.into()).collect()))
        }

    }

    fn get_decoder<P: AsRef<Path>>(&self, archive: P) -> io::Result<GzDecoder<File>> {
        let archive = File::open(archive)?;
        Ok(GzDecoder::new(archive))
    }
}

fn main() {
    let builder = FashionMNISTBuilder::new();
    let mnist = builder.data_home("fashion_mnist").get_data().unwrap();

}