use std::fs::File;
use std::path::Path;
use rawloader::RawImage;


#[test]
fn load_dng_v6_6() {
    let loader = rawloader::RawLoader::new();
    let mut file = File::open(Path::new("tests/resources/set_1/x_x.dng")).unwrap();
    let image: RawImage = loader.decode(&mut file, false).unwrap();
    // println!("Loaded Image Dims: [{}, {}]", data.width, data.height);

    if let rawloader::RawImageData::Integer(x) = image.data {
        x.as_slice().chunks(3).take(5).for_each(|x| {
            println!("{}", x.iter().map(|x| {
                format!("{:b}", x)
            }).collect::<Vec<String>>().join(" ")
        )});
    };
}


