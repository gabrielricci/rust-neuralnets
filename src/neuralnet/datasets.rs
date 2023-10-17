use ndarray::prelude::*;
use polars::prelude::*;
use std::path::PathBuf;

pub fn dataframe_from_csv(filepath: PathBuf) -> PolarsResult<(DataFrame, DataFrame)> {
    let data = CsvReader::from_path(filepath)?.has_header(true).finish()?;

    let training_dataset = data.drop("label")?;
    let training_labels = data.select(["label"])?;

    Ok((training_dataset, training_labels))
}

pub fn array_from_dataframe(df: &DataFrame) -> Array2<f32> {
    df.to_ndarray::<Float32Type>(IndexOrder::C)
        .unwrap()
        .reversed_axes()
}
