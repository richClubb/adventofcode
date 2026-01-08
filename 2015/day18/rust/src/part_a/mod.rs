use std::fs::File;
use std::io::{BufRead, BufReader};

// find some way to get the pointers to the other objects

#[derive(Clone, Debug)]
struct Pixel<'a>{
    curr_value: bool,
    buffer: bool,
    neighbors: Vec<&'a Pixel<'a>>
}

impl Pixel<'_>{

    pub fn new(value: bool) -> Self {
        Pixel { curr_value: value, buffer: value, neighbors: Vec::new()}
    }

    pub fn add_neighbour<'a>(&mut self, neighbour: &Pixel<'_> ){
        
        // this needs to be an actual reference
        //self.neighbors.push(neighbour);
    }

    pub fn calc_next_value(&self) -> bool {

        if self.buffer != self.curr_value {
            return true;
        }

        return false
    }

    pub fn update(&self) {

    }

}


struct PixelMap<'a> {
    pixels: Vec<Vec<Pixel<'a>>>,
    x_size: usize,
    y_size: usize,
}

impl PixelMap<'_> {
    pub fn new<'a>(x_size: usize, y_size: usize) -> Self {
        
        return PixelMap{ pixels: Vec::new(), x_size: x_size, y_size: y_size };
    }

    pub fn add_pixels<'a>(&mut self, input: Vec<Vec<char>>) {


        // load in pixels
        for line in input {
            let mut pixel_line: Vec<Pixel> = Vec::new();
            for entry in line {
                let val = if entry == '#' { true } else { false };
                pixel_line.push(Pixel::new(val));
            }
            self.pixels.push(pixel_line);
        }

        // set up neighbours
        for y_index in 0..self.y_size{
            for x_index in 0..self.x_size{
                
                // println!("For pixel '{y_index},{x_index}'");
                for y_pixel in -1..=1 as isize {
                    for x_pixel in -1..=1 as isize {
                        let y_offset = y_index as isize - y_pixel;
                        let x_offset = x_index as isize - x_pixel;

                        if (y_pixel == 0) && 
                           (x_pixel == 0) 
                        {
                            continue;
                        }

                        if (y_offset < 0) || 
                           (y_offset >= self.y_size as isize) ||
                           (x_offset < 0) || 
                           (x_offset >= self.x_size as isize) 
                        {
                            continue;
                        }

                        // println!("  offset '{y_offset},{x_offset}'");

                        let base_pixel = &self.pixels[x_index as usize][y_index as usize];
                        let neighbour_pixel = &self.pixels[y_offset as usize][x_offset as usize];

                        base_pixel.add_neighbour(&neighbour_pixel);
                        // self.pixels.[x_index][y_index]
                        //     .neighbors
                        //     .push(&self.pixels.[y_offset as usize][x_offset as usize]);

                    }
                }
            }
        }
    }

    pub fn print(&self) {
        for y_index in 0..self.y_size {
            self.pixels[y_index].iter().for_each(|item| {
                    let val = if item.curr_value { "#" } else { "." }; 
                    print!("{val} ");
                }
            );
            println!("");
        }
    }

    pub fn step(&self) {

    }

}

pub fn part_a(path: &String, steps: &usize)
{
    println!("Part A");

    let file: File = File::open(path).expect("Could not open file");
    let buf_reader:BufReader<File> = BufReader::new(file);

    let mut pixel_chars: Vec<Vec<char>> = Vec::new();

    buf_reader.lines().for_each(|line| {
            let line = line.unwrap();
            let line = line.trim();
            pixel_chars.push(line.chars().collect())
        }
    );
    
    let y_size = pixel_chars.len();
    let x_size = pixel_chars[0].len();

    let mut pixel_map  = PixelMap::new(x_size, y_size);
    pixel_map.add_pixels(pixel_chars);

    pixel_map.print();
}