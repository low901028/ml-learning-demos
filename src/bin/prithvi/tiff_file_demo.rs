use std::any::{Any, TypeId};
use std::fmt::Debug;
use tiff::decoder::{Decoder, DecodingBuffer, DecodingResult};
use tiff::tags::Tag;
use tiff::ColorType;

use std::fs::File;
use std::io::{BufReader, Read};
use std::path::PathBuf;
use gdal::{Dataset, Metadata};
use gdal::vector::LayerAccess;
use image::GenericImageView;
use tiff::decoder::ifd::Value::Ascii;
use tiff::encoder::TiffValue;
use tiff::tags::Tag::{GeoDoubleParamsTag, ModelPixelScaleTag, RowsPerStrip, StripByteCounts, StripOffsets};

const TEST_IMAGE_DIR: &str = "/Users/dalan/rustspace/ml-learning-demos/prithvi-model/tifs/alaska";

fn read_u32<T: Read>(reader: &mut T) -> std::io::Result<u32> {
    use byteorder::ReadBytesExt;
    // reader.read_u32::<byteorder::BigEndian>()
    reader.read_u32::<byteorder::LittleEndian>()
}

mod tests {
    use std::fs::File;
    use std::path::PathBuf;
    use gdal::Metadata;
    use tiff::decoder::{Decoder, DecodingResult};
    use tiff::encoder::TiffValue;
    use tiff::tags::Tag;
    use tiff::tags::Tag::{RowsPerStrip, StripOffsets};
    use crate::tiff_file_demo::TEST_IMAGE_DIR;
    use candle_core::{shape, Device, Tensor};

    #[test]
    fn test_geo_tiff() {
        let filenames = ["Alaska_HLS.S30.T06VUN.2020305T212629.v2.0_cropped.tif",
            "Alaska_HLS.S30.T06VUN.2021044T212601.v2.0_cropped.tif",
            "Alaska_HLS.S30.T06VUN.2021067T213531.v2.0_cropped.tif",
            "Alaska_HLS.S30.T06VUN.2021067T213531.v2.0_cropped_v2.tif",
        ];
        for filename in filenames.iter() {
            let path = PathBuf::from(TEST_IMAGE_DIR).join(filename);
            let img_file = File::open(path).expect("Cannot find test image!");
            let mut decoder = Decoder::new(img_file).expect("Cannot create decoder");
            decoder = decoder.with_limits(tiff::decoder::Limits::unlimited());

            assert_eq!(
                decoder.dimensions().expect("Cannot get dimensions"),
                (560, 448)
            );
            // 验证颜色类型
            // let color_type = decoder.colortype().expect("Cannot get colortype");
            // println!("color type: {:?}", color_type);

            // assert_eq!(
            //     decoder
            //         .get_tag_u64(Tag::StripOffsets)
            //         .expect("Cannot get StripOffsets"),
            //     418
            // );
            assert_eq!(
                decoder
                    .get_tag_u64(Tag::RowsPerStrip)
                    .expect("Cannot get RowsPerStrip"),
                1
            );
            // assert_eq!(
            //     decoder
            //         .get_tag_u64(Tag::StripByteCounts)
            //         .expect("Cannot get StripByteCounts"),
            //     1000
            // );
            assert_eq!(
                decoder
                    .get_tag(Tag::ModelPixelScaleTag)
                    .expect("Cannot get pixel scale")
                    .into_f64_vec()
                    .expect("Cannot get pixel scale"),
                vec![30.0, 30.0, 0.0]
            );
            let data = decoder.read_image().unwrap() else {
                panic!("Cannot read band data")
            };
            println!("decoding result:{:?}", data)
        }
    }

    #[test]
    fn test_big_tiff() {
        let filenames = ["Alaska_HLS.S30.T06VUN.2020305T212629.v2.0_cropped.tif",
            "Alaska_HLS.S30.T06VUN.2021044T212601.v2.0_cropped.tif",
            "Alaska_HLS.S30.T06VUN.2021067T213531.v2.0_cropped.tif",
            "Alaska_HLS.S30.T06VUN.2021067T213531.v2.0_cropped_v2.tif",
        ];
        for filename in filenames.iter() {
            let path = PathBuf::from(TEST_IMAGE_DIR).join(filename);
            let img_file = File::open(path).unwrap();

            let mut decoder = Decoder::new(img_file).expect("Cannot create decoder");
            decoder = decoder.with_limits(tiff::decoder::Limits::unlimited());

            let _dimensions = decoder.dimensions();
            println!("===dimensions={:?}", _dimensions);
            let _colortype = decoder.colortype().expect("Cannot get colortype");
            println!("===colortype={:?}", _colortype);
            let _tag_offsets = decoder.get_tag(StripOffsets).expect("Cannot get StripOffsets");
            // println!("===strip offsets={:?}", _tag_offsets);
            let _row_per_strip = decoder
                .get_tag_u64(RowsPerStrip)
                .expect("Cannot get RowsPerStrip");
            println!("row per strip={:?}", _row_per_strip);
            // let _strip_byte_counts = decoder
            //     .get_tag_u64(Tag::StripByteCounts)
            //     .expect("Cannot get StripByteCounts");
            // println!("===strip byte counts={:?}", _strip_byte_counts);
            let _model_pixel_scale = decoder
                .get_tag(Tag::ModelPixelScaleTag)
                .expect("Cannot get pixel scale")
                .into_f64_vec()
                .expect("Cannot get pixel scale");
            println!("model pixel scale: {:?}", _model_pixel_scale);

            let DecodingResult::I16(data) = decoder.read_image().unwrap() else {
                panic!("Cannot read band data")
            };
            println!("geo tiff file length={:?}", &data.len());

            let datas = data.data().to_vec();
            println!("cow data tiff file length={:?}", &datas.len());

            let chunk_dimensions = decoder.chunk_dimensions();
            println!("chunk dimensions={:?}", chunk_dimensions);

            println!("===end===");
        }
    }

    // #[test]
    // fn test_geotiff_file_demo() {
    //     use std::fs::File;
    //     use std::path::PathBuf;
    //     use geotiff::GeoTiff;
    //     let tiff_file = "/Users/dalan/rustspace/ml-learning-demos/prithvi-model/tifs/alaska/Alaska_HLS.S30.T06VUN.2020305T212629.v2.0_cropped.tif";
    //     let geotiff = GeoTiff::read(File::open(tiff_file)
    //         .expect("File io error"))
    //         .expect("file io error...");
    //
    // }

    #[test]
    fn test_image_tiff_demo() {
        use image::{DynamicImage, GenericImageView, ImageFormat, ImageResult};
        use std::fs::File;
        use std::io::BufReader;
        fn read_geotiff(file_path: &str) -> ImageResult<DynamicImage> {
            let file = File::open(file_path)?;
            let reader = BufReader::new(file);
            image::load(reader, ImageFormat::Tiff)
        }
        println!("Hello, world!");
        let tiff_file = "/Users/dalan/rustspace/ml-learning-demos/prithvi-model/tifs/alaska/Alaska_HLS.S30.T06VUN.2020305T212629.v2.0_cropped.tif";
        match read_geotiff(tiff_file) {
            Ok(image) => {
                println!("Image dimensions: {:?}", image.dimensions());
                // 进一步处理图像数据
            }
            Err(e) => {
                eprintln!("Failed to read GeoTIFF file: {}", e);
            }
        }

    }
    //
    #[test]
    fn test_georust_tiff_demo() -> candle_core::Result<()>{
        use gdal::Dataset;
        let path = "/Users/dalan/rustspace/ml-learning-demos/prithvi-model/tifs/alaska/Alaska_HLS.S30.T06VUN.2020305T212629.v2.0_cropped.tif";
        let ds = Dataset::open(path).unwrap();

        ds.rasterbands().into_iter().for_each(|x| match x {
            Ok(band) => {
                let array = band.read_band_as::<u8>().unwrap().to_array().unwrap();
                println!("array shape={:?}", array.shape());
            },
            Err(e) => {
                println!("{}", e);
             }
        });
        let rasterband = ds.rasterband(1).unwrap();
        let buffers = rasterband.read_band_as::<u8>().unwrap().to_array().unwrap();
        let datas = buffers.iter().map(|x| *x).collect::<Vec<u8>>();

        let tensor =  Tensor::from_vec(datas, buffers.shape(), &Device::Cpu)?;
        println!("tensor shape={:?}, data={}", tensor.shape(), tensor);

        println!("Driver: {}", ds.driver().long_name());
        let layer = ds.layer(0);

        // for layer in layer.iter() {
        //     if let Some(feature) = layer.feature(0) {
        //         if let Some(geometry) = feature.geometry() {
        //             if let Ok(wkt_text) = geometry.wkt() {
        //                 println!("{}", wkt_text);
        //             }
        //         }
        //     }
        // }

        println!("raster_size={:?}, projection={:?}",
                 ds.raster_size(),
                 ds.projection());

        ds.metadata().into_iter().for_each(|entry| {
            println!("key={:?}, value={:?}", entry.key, entry.value);
        });
        Ok(())
    }

    #[test]
    fn test_tiff_file_demo() -> tiff::TiffResult<()>{
        #[derive(Debug, serde::Serialize, serde::Deserialize, Default, Clone)]
        struct GDALMetadata {
            resolution: usize,
            spec: String,
            resolutions_xy:(usize, usize),
            bands: Vec<usize>,
        }

        let filenames = ["Alaska_HLS.S30.T06VUN.2020305T212629.v2.0_cropped.tif",
            "Alaska_HLS.S30.T06VUN.2021044T212601.v2.0_cropped.tif",
            "Alaska_HLS.S30.T06VUN.2021067T213531.v2.0_cropped.tif",
            "Alaska_HLS.S30.T06VUN.2021067T213531.v2.0_cropped_v2.tif",
        ];
        for filename in filenames.iter() {
            let path = PathBuf::from(TEST_IMAGE_DIR).join(filename);
            let img_file = File::open(path).unwrap();

            let mut decoder = Decoder::new(img_file).expect("Cannot create decoder");
            decoder = decoder.with_limits(tiff::decoder::Limits::unlimited());
            match decoder.get_tag(Tag::Unknown(42112)) {
                Ok(tag) => {
                    println!("Tag::Unknown(42112), value={:?}", tag.into_string());
                },
                Err(err) => {
                    println!("err={:?}", err);
                }
            }
            // decoder.tag_iter().for_each(|tag| {
            //     let (tag, value) = tag.unwrap();
            //     println!("tag={:?}, value={:?}", tag, value);
            // });

            let mut datas = decoder.read_image()?;
            println!("========end ============");
        }

        Ok(())
    }

    #[test]
    fn test_serde_ascii_demo() -> serde_json::Result<()>{
        let content = r#"Ascii("<GDALMetadata>\n  <Item name=\"resolution\">30</Item>\n  <Item name=\"spec\">RasterSpec(epsg=32605, bounds=(655020, 6775380, 671820, 6788820), resolutions_xy=(30, 30))</Item>\n  <Item name=\"DESCRIPTION\" sample=\"0\" role=\"description\">bands</Item>\n  <Item name=\"DESCRIPTION\" sample=\"1\" role=\"description\">bands</Item>\n  <Item name=\"DESCRIPTION\" sample=\"2\" role=\"description\">bands</Item>\n  <Item name=\"DESCRIPTION\" sample=\"3\" role=\"description\">bands</Item>\n  <Item name=\"DESCRIPTION\" sample=\"4\" role=\"description\">bands</Item>\n  <Item name=\"DESCRIPTION\" sample=\"5\" role=\"description\">bands</Item>\n</GDALMetadata>\n")"#;
        let ascii_str: serde_json::Value = serde_json::from_slice(content.as_bytes())?;
        println!("=====");

        Ok(())
    }
}
