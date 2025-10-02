pub struct SeedRange{
    pub start: u64,
    pub end: u64,
    pub size: u64
}

impl SeedRange{

    pub fn new(start: u64, size: u64) -> SeedRange {
        return SeedRange{start: start, end: start + size, size: size}
    }
}

fn split_range(input_range: SeedRange) -> SeedRange {

    return SeedRange{start: 0, end: 0, size: 0}
}


fn main() {
    println!("Hello, world!");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn map_seed() {

    }

}