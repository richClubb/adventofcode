use std::fs::File;
use std::io::{BufRead, BufReader};

// find some way to get the pointers to the other objects

struct PixelMap {
    pixels: Vec<Vec<bool>>,
}

impl PixelMap {
    pub fn new(input: Vec<Vec<char>>) -> Self {

        let mut pixels = Vec::new();

        // load in pixels
        for line in input {
            let mut pixel_line: Vec<bool> = Vec::new();
            for entry in line {
                let val = if entry == '#' { true } else { false };
                pixel_line.push(val);
            }
            pixels.push(pixel_line);
        }

        let y_size = pixels.len();
        let x_size = pixels[0].len();

        pixels[0][0] = true;
        pixels[0][x_size-1] = true;
        pixels[y_size - 1][0] = true;
        pixels[y_size - 1][x_size - 1] = true;

        return PixelMap{ pixels: pixels };
    }

    pub fn print(&self) -> usize {

        let mut light_count: usize = 0;
        let y_size = self.pixels.len();

        for y_index in 0..y_size {
            self.pixels[y_index].iter().for_each(|item| {
                    let val = if item == &true { 
                        light_count += 1;
                        "#" 
                    } 
                    else { 
                        "." 
                    }; 
                    print!("{val} ");
                }
            );
            println!("");
        }

        return light_count;
    }

    pub fn step(&self) -> PixelMap {

        let mut pixels: Vec<Vec<bool>> = Vec::new();

        let y_size = self.pixels.len();
        let x_size = self.pixels[0].len();

        for y_index in 0..y_size {

            let mut pixel_line: Vec<bool> = Vec::new();

            for x_index in 0..x_size {

                let mut neighbour_value: usize = 0;

                if ((y_index == 0)            && (x_index == 0)) ||
                   ((y_index == 0)            && (x_index == (x_size - 1))) ||
                   ((y_index == (y_size - 1)) && (x_index == 0)) ||
                   ((y_index == (y_size - 1)) && (x_index == (x_size - 1))) 
                {
                    pixel_line.push(true);
                    continue;
                }

                // println!("Pixel {y_index},{x_index} is {}", self.pixels[y_index][x_index]);
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
                           (y_offset >= y_size as isize) ||
                           (x_offset < 0) || 
                           (x_offset >= x_size as isize) 
                        {
                            continue;
                        }

                        let neighbour_pixel = self.pixels[y_offset as usize][x_offset as usize];

                        if neighbour_pixel {
                            // println!("  {y_index},{x_index} pixel {y_offset},{x_offset} is on");
                            neighbour_value += 1;
                        }
                        else {
                            // println!("  {y_index},{x_index} pixel {y_offset},{x_offset} is off");
                        }
                    }  
                }

                // println!("Pixel {y_index},{x_index}: {neighbour_value}");

                if self.pixels[y_index][x_index] {
                    let val = if (neighbour_value >= 2) && (neighbour_value <= 3) {
                        true
                    }
                    else {
                        false
                    };
                    // println!("  pixel on {y_index},{x_index} pushing {val}");
                    pixel_line.push(val);
                }
                else {
                    let val = if neighbour_value == 3 {
                        true
                    }
                    else {
                        false
                    };
                    // println!("  pixel off {y_index},{x_index} pushing {val}");
                    pixel_line.push(val);
                }   
            }

            pixels.push(pixel_line);
        }

        return PixelMap{ pixels: pixels };
    }

}

pub fn part_b(path: &String, steps: &usize)
{
    println!("Part B");

    let file: File = File::open(path).expect("Could not open file");
    let buf_reader:BufReader<File> = BufReader::new(file);

    let mut pixel_chars: Vec<Vec<char>> = Vec::new();

    buf_reader.lines().for_each(|line| {
            let line = line.unwrap();
            let line = line.trim();
            pixel_chars.push(line.chars().collect())
        }
    );

    let mut pixel_map  = PixelMap::new(pixel_chars);


    println!("Initial ");
    pixel_map.print();
    

    for step in 0..steps.clone() {
        println!("\nStep {}", step + 1);
        pixel_map = pixel_map.step();
        let val = pixel_map.print();
        println!("{val} lights are on");         
    }
}