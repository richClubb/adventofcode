using System.CommandLine;
using System.CommandLine.Parsing;

using part_a;
using part_b;

namespace day5;

class Program
{
    static int Main(string[] args)
    {
        Option<string> file_path = new("--file")
        {
            Description = "The file to read and display on the console."
        };

        Option<string> run_type = new("--run")
        {
            Description = "The run to perform."
        };

        RootCommand rootCommand = new("Day 5");
        rootCommand.Options.Add(file_path);
        rootCommand.Options.Add(run_type);

        ParseResult parseResult = rootCommand.Parse(args);
        if (parseResult.Errors.Count != 0)
        {
            foreach (ParseError parseError in parseResult.Errors)
            {
                Console.Error.WriteLine(parseError.Message);
            }    
            return 1;
        }

        if (parseResult.GetValue(file_path) == null)
        {
            Console.Error.WriteLine("No file path specified");
            return 1;
        }

        if (parseResult.GetValue(run_type) == null)
        {
            Console.Error.WriteLine("No run type pecified");
            return 1;
        }

        if (parseResult.GetValue(run_type) == "part_a")
        {
            var result = PartA.Run(parseResult.GetValue(file_path));
            Console.WriteLine("Part A result is: {0}", result);
        }

        if (parseResult.GetValue(run_type) == "part_b")
        {
            var result = PartB.Run(parseResult.GetValue(file_path));
            Console.WriteLine("Part B result is: {0}", result);
        }
        return 0;
    }
}
