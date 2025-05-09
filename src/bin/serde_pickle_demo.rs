use anyhow::{Context, Result};
use serde_pickle::Value;
use std::fs::File;
use std::io::BufReader;

pub fn read_pt_file(file_path: &str) -> Result<()> {
    // let content = candle_core::pickle::read_all(file_path)?;
    // for (key, tensor) in content {
    //     println!("content key={:?}, tensor={}", key, tensor);
    // }
    let tensors =
        candle_core::pickle::PthTensors::new(file_path, None)
            .unwrap();
    Ok(())
}

pub fn read_pickle_file(file_path: &str) -> Result<Value> {
    let file = File::open(file_path)
        .with_context(|| format!("Failed to open file: {}", file_path))?;
    let mut reader = BufReader::new(file);

    
    let content = serde_pickle::value_from_reader(&mut reader, serde_pickle::DeOptions::new())
        .with_context(|| format!("Failed to parse pickle file: {}", file_path));
    
    serde_pickle::from_reader(&mut reader, serde_pickle::DeOptions::new())
        .with_context(|| format!("Failed to parse pickle file: {}", file_path))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_pickle::Value;
    use std::collections::HashMap;

    fn create_test_pickle() {
        // Python code to generate test pickle:
        // import pickle
        // test_data = {
        //     "name": "Alice",
        //     "age": 30,
        //     "scores": [85, 92, 78],
        //     "metadata": {"id": 123, "valid": True}
        // }
        // with open("tests/test_data.pkl", "wb") as f:
        //     pickle.dump(test_data, f)

        // This binary data represents the above Python code's output
        let pickle_data: [u8; 88] = [
            0x80, 0x04, 0x95, 0x50, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x7d, 0x94, 0x28, 0x8c,
            0x04, 0x6e, 0x61, 0x6d, 0x65, 0x94, 0x8c, 0x05, 0x41, 0x6c, 0x69, 0x63, 0x65, 0x94, 0x8c,
            0x03, 0x61, 0x67, 0x65, 0x94, 0x4e, 0x1e, 0x8c, 0x06, 0x73, 0x63, 0x6f, 0x72, 0x65, 0x73,
            0x94, 0x5d, 0x94, 0x28, 0x4e, 0x55, 0x4e, 0x5e, 0x4e, 0x4e, 0x65, 0x8c, 0x08, 0x6d, 0x65,
            0x74, 0x61, 0x64, 0x61, 0x74, 0x61, 0x94, 0x7d, 0x94, 0x28, 0x8c, 0x02, 0x69, 0x64, 0x94,
            0x4e, 0x7b, 0x8c, 0x05, 0x76, 0x61, 0x6c, 0x69, 0x64, 0x94, 0x88, 0x75, 0x75,
        ];

        std::fs::create_dir_all("tests").unwrap();
        std::fs::write("tests/test_data.pkl", &pickle_data).unwrap();
    }
    
    #[test]
    fn test_file_not_found() {
        let result = read_pickle_file("nonexistent.pkl");
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("Failed to open file"));
    }
}

fn main() {
    // let path = "/Users/dalan/rustspace/ml-learning-demos/mnist/pkl/sample_weight.pkl";
    // let pickle = read_pickle_file(path).unwrap();
    // println!("{:?}", pickle);
    
    let path = "/Users/dalan/rustspace/ml-learning-demos/mnist/pkl/sample_weight.pt";
    read_pt_file(path).unwrap();
}