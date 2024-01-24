use clap::Parser;

#[derive(Parser)]
struct Cli {
    path: String,
    run: String,
}


fn main() {
    let args = Cli::parse();

    println!("path: {:?}, run: {:?}", args.path, args.run)
}
