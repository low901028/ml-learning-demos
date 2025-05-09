use std::fs::File;
use candle_core::Device;
use candle_core::quantized::QMatMul::Tensor;
use geotiff::GeoTiff;
use image::{buffer, DynamicImage, GenericImageView, GrayImage, ImageFormat, ImageReader, Rgb, RgbImage};
use image::imageops::FilterType;
use parquet::data_type::AsBytes;
use plotters::backend::{BGRXPixel, BitMapBackend, PixelFormat, RGBPixel};
use plotters::chart::ChartBuilder;
use plotters::element::BitMapElement;
use plotters::prelude::{DrawingBackend, IntoDrawingArea, IntoLinspace, RED, WHITE};

fn plotter_bitmap_demo() -> Result<(), Box<dyn std::error::Error>> {
    use plotters::prelude::*;
    use std::fs::File;
    use std::io::BufReader;

    const OUT_FILE_NAME: &str = "plotters-doc-data-bitmap.png";
    let root = BitMapBackend::new(OUT_FILE_NAME, (1024, 768)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption("Bitmap Example", ("sans-serif", 30))
        .margin(5)
        .set_label_area_size(LabelAreaPosition::Left, 40)
        .set_label_area_size(LabelAreaPosition::Bottom, 40)
        .build_cartesian_2d(0.0..1.0, 0.0..1.0)?;

    chart.configure_mesh().disable_mesh().draw()?;

    let (w, h) = chart.plotting_area().dim_in_pixel();
    let image = image::load(
        BufReader::new(
            File::open("basic-normal-dist.png").map_err(|e| {
                eprintln!("Unable to open file plotters-doc-data.png, please make sure you have clone this repo with --recursive");
                e
            })?),
        ImageFormat::Png,
    )?
        .resize_exact(w - w / 10, h - h / 10, FilterType::Nearest);

    // let elem = BitMapElement::from(((0.05, 0.95), image).into());
    let elem = BitMapElement::with_owned_buffer((0.05, 0.95), (w - w / 10, h - h / 10), image.into_bytes());
    // let elem: BitMapElement<_,_> = ((0.05, 0.95), image).into();
    //
    chart.draw_series(std::iter::once(elem.unwrap()))?;
    // To avoid the IO failure being ignored silently, we manually call the present function
    // root.present().expect("Unable to write result to file, please make sure 'plotters-doc-data' dir exists under current dir");
    root.present().unwrap();
    println!("Result has been saved to {}", OUT_FILE_NAME);
    Ok(())
}

fn draw_bitmap_from_vector() {
    use std::io::Cursor;
    use image::ImageFormat;

    let buffer = vec![
        vec![
        vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 2, 0, 0, 0, 0, 3, 0, 38, 55, 0],
        vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 3, 2, 0, 0, 53, 146, 0],
        vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 107, 182, 0, 0, 0, 0, 0, 0, 9, 106, 173, 0],
        vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 174, 226, 199, 13, 38, 70, 74, 167, 225, 237, 117, 0],
        vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 62, 220, 194, 211, 226, 255, 235, 237, 224, 204, 224, 98, 0],
        vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 196, 214, 208, 213, 218, 215, 220, 218, 208, 204, 225, 69, 0],
        vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 80, 222, 201, 208, 220, 224, 222, 220, 227, 210, 202, 217, 32, 0],
        vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 240, 209, 203, 210, 223, 222, 225, 222, 230, 212, 203, 218, 61, 0],
        vec![0, 0, 0, 0, 0, 1, 1, 0, 2, 0, 2, 0, 0, 210, 216, 206, 203, 213, 225, 221, 224, 225, 225, 220, 215, 239, 58, 0],
        vec![0, 0, 0, 0, 0, 1, 0, 0, 0, 3, 0, 0, 180, 211, 200, 202, 208, 221, 221, 221, 224, 223, 221, 224, 209, 234, 116, 0],
        vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 190, 206, 194, 199, 201, 210, 225, 215, 219, 222, 221, 223, 223, 203, 226, 224, 0],
        vec![0, 0, 0, 1, 2, 1, 0, 0, 0, 86, 215, 202, 193, 202, 203, 202, 224, 211, 211, 215, 218, 221, 229, 220, 216, 216, 254, 22],
        vec![1, 6, 4, 0, 0, 0, 0, 32, 174, 231, 187, 187, 208, 201, 205, 205, 211, 215, 214, 217, 219, 215, 214, 213, 213, 212, 230, 82],
        vec![0, 0, 0, 0, 0, 36, 150, 209, 189, 163, 176, 199, 199, 202, 202, 198, 197, 203, 205, 207, 209, 205, 212, 214, 215, 214, 227, 109],
        vec![0, 34, 146, 199, 201, 226, 193, 181, 189, 184, 194, 199, 194, 200, 204, 209, 214, 212, 206, 206, 208, 208, 216, 216, 219, 214, 231, 81],
        vec![31, 197, 198, 193, 191, 188, 185, 194, 196, 189, 190, 191, 196, 206, 207, 220, 232, 236, 239, 242, 249, 252, 252, 246, 221, 213, 248, 37],
        vec![56, 197, 190, 212, 216, 219, 212, 210, 209, 205, 211, 219, 220, 252, 253, 247, 226, 203, 186, 167, 150, 140, 133, 123, 113, 98, 99, 10],
        vec![23, 89, 93, 114, 135, 148, 167, 182, 192, 210, 218, 210, 192, 158, 128, 96, 81, 81, 82, 82, 82, 84, 88, 92, 93, 94, 107, 33],
        vec![11, 62, 74, 75, 73, 74, 77, 80, 81, 87, 83, 80, 80, 79, 90, 102, 105, 106, 108, 110, 110, 109, 109, 109, 105, 103, 108, 36],
        vec![0, 0, 0, 0, 8, 12, 17, 29, 41, 53, 56, 65, 78, 77, 75, 74, 76, 76, 69, 61, 58, 60, 56, 52, 43, 40, 27, 0],
        vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            ]
    ];

    let mut raw_data = buffer.into_iter().flatten().flatten().collect::<Vec<_>>();;
    let rgbimage = GrayImage::from_vec(28,28, raw_data);
    let buffer = DynamicImage::from(rgbimage.unwrap()).to_rgb8().to_vec();


    let bitmap = BitMapElement::with_owned_buffer((10.0, 18.0), (28, 28), buffer);

    const OUT_FILE_NAME: &str = "label_tag_blit-bitmap.png";
    // use plotters::prelude::*;

    let root = BitMapBackend::new(OUT_FILE_NAME, (1024, 768)).into_drawing_area();
    root.fill(&WHITE).unwrap();
    root.split_evenly((2,9));

    let mut chart = ChartBuilder::on(&root)
        .caption("Bitmap Example", ("sans-serif", 30))
        .margin(5)
        .build_cartesian_2d(0.0..80.0, 0.0..80.0).unwrap();

    chart.configure_mesh().disable_mesh().draw().unwrap();

    if let Some(elem) = bitmap {
        println!("drawing bitmap...");
        chart.draw_series(std::iter::once(elem)).unwrap();
    }
}

fn vector_nested_demo() {
    let v = vec![vec![vec![1,2,3], vec![1,2,3]], vec![vec![4,5,6]]];
    let datas = v.into_iter().flatten().flatten().collect::<Vec<i32>>();
    println!("{:?}", datas);
}

// fn gdal_geotiff_demo() {
//     use gdal::raster::RasterBand;
//     use gdal::{Dataset, Metadata};
//     use std::path::Path;
//
//     let dataset = Dataset::open("/Users/dalan/rustspace/ml-learning-demos/prithvi-model/tifs/alaska/Alaska_HLS.S30.T06VUN.2020305T212629.v2.0_cropped.tif").unwrap();
//     println!("dataset description: {:?}", dataset.description());
// }

fn geotiff_demo() {
    use geotiff::GeoTiff;
    let tiff_file = "/Users/dalan/rustspace/ml-learning-demos/prithvi-model/tifs/alaska/Alaska_HLS.S30.T06VUN.2020305T212629.v2.0_cropped.tif";
    let path = std::path::PathBuf::from(tiff_file);
    let geo = GeoTiff::read(std::fs::File::open(&path).expect("File I/O error")).expect("File I/O error");
    println!("{:?}", geo);
}

fn test_geotiff_demo() {
    use std::fs::File;
    use std::path::PathBuf;
    use geotiff::GeoTiff;
    use image::ImageReader;
    use geotiff::GeoKeyDirectory;
    use geotiff::RasterType;
    fn read_geotiff(path: &str) -> std::result::Result<(), Box<dyn std::error::Error>> {
        let img = ImageReader::open(path)?.decode()?;
        println!("Image dimensions: {:?}", img.dimensions());

        let file = File::open(path)?;
        let mut reader = GeoTiff::read(file)?;
        println!("raster size= height={:?} weight={:?}", reader.raster_height, reader.raster_width);
        println!("samples={:?}", reader.num_samples);
        println!("geo keys={:?}", reader.geo_key_directory);
        Ok(())
    }

    let tiff_file = "/Users/dalan/rustspace/ml-learning-demos/prithvi-model/tifs/alaska/Alaska_HLS.S30.T06VUN.2020305T212629.v2.0_cropped.tif";
    read_geotiff(tiff_file).unwrap();

}
fn gettiff2_demo() {
    use log::{error, info};
    use std::{collections::BTreeMap, fs, path::Path, time::Instant};
    use tiff2::{decoder::Decoder, error::TiffResult};

    fn test_tiffs() {
        let tiff_file = "/Users/dalan/rustspace/ml-learning-demos/prithvi-model/tifs/alaska/Alaska_HLS.S30.T06VUN.2020305T212629.v2.0_cropped.tif";
        match GeoTiff::read(
            File::open(tiff_file).expect("could not parse path"),
        ) {
            Ok(x) => {
                println!("great success! {:?}", x);
            }
            Err(e) => {
                println!("Fail! {:?}", e);
                panic!("Failed at {:?}", e)
            }
        };
    }

    test_tiffs();
}

fn test_to_scalar_demo() -> candle_core::Result<()>{
    let device = Device::cuda_if_available(0)?;
    let datas = candle_core::Tensor::new(&[6u32], &device)?;
    println!("{},shape rank={:?}", datas, datas.shape().rank());
    println!("{:?}", datas.to_scalar::<u32>());
    
    
    Ok(())
}
fn main() {
    println!("Hello, world!");
    test_to_scalar_demo().unwrap();
    // gettiff2_demo();
    // test_geotiff_demo();
    // geotiff_demo();
    // gdal_geotiff_demo();
    // vector_nested_demo();
    // draw_bitmap_from_vector();
    // plotter_bitmap_demo().unwrap();
}
