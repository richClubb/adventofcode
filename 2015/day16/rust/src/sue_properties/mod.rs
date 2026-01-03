use regex::Regex;

#[derive(Debug)]
pub struct SueProperties {
    pub children: Option<usize>,
    pub cats: Option<usize>,
    pub samoyeds: Option<usize>,
    pub pomeranians: Option<usize>,
    pub akitas: Option<usize>,
    pub vizslas: Option<usize>,
    pub goldfish: Option<usize>,
    pub trees: Option<usize>,
    pub cars: Option<usize>,
    pub perfumes: Option<usize>,
}

impl SueProperties {

    pub fn new(input_string: &String) -> Self {

        let input_vec = input_string.split(",");

        let item_re = Regex::new(r"(?<item>[a-z]{1,})\:\s(?<qty>[0-9]{1,})").unwrap();

        let mut sue = SueProperties { 
            children: None, 
            cats: None, 
            samoyeds: None, 
            pomeranians: None, 
            akitas: None, 
            vizslas: None, 
            goldfish: None, 
            trees: None, 
            cars: None, 
            perfumes: None 
        };

        for item in input_vec {

            let (item, qty) = item_re.captures(item).map(|caps| {
                    let item = caps.name("item").unwrap().as_str();
                    let qty = caps.name("qty").unwrap().as_str().parse::<usize>().unwrap();

                    (item, qty)
                }
            ).unwrap();

            match item {
                "children" => sue.children = Some(qty), 
                "cats" => sue.cats = Some(qty),
                "samoyeds" => sue.samoyeds = Some(qty),
                "pomeranians" => sue.pomeranians = Some(qty),
                "akitas" => sue.akitas = Some(qty),
                "vizslas" => sue.vizslas = Some(qty),
                "goldfish" => sue.goldfish = Some(qty),
                "trees" => sue.trees = Some(qty),
                "cars" => sue.cars = Some(qty),
                "perfumes" => sue.perfumes = Some(qty),
                _ => (),
            };
        };

        return sue;
    }

}