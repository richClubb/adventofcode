package com.mycompany.app;

import org.apache.commons.cli.*;

/**
 * Hello world!
 */
public class Day5 {
    public static void main(String[] args) {
        Options options = new Options();

        options.addOption("f", "file", true, "Input file");
        options.addOption("r", "run", true, "Run type");

        CommandLineParser parser = new DefaultParser();
        CommandLine cmd = null;
        
        try {
            cmd = parser.parse(options, args);
        }
        catch (Exception ex)
        {
            System.out.println("Failed to parse command line parameters: " + ex.getMessage());
            return;
        }

        if (!cmd.hasOption("f") || !cmd.hasOption("r"))
        {
            System.err.println("Must specimy both file ('-f' / '--file') and run ('-r', '--run')");
            return;
        }

        String run_type = cmd.getOptionValue("r");

        if (run_type.compareTo("part_a") == 0)
        {
            try 
            {
                long result = PartA.Run(cmd.getOptionValue("f"));
                System.out.printf("Part A result: %d\n", result);
            }
            catch (Exception ex)
            {
                System.err.println("Failed to execute part a");
            }
        }
        else if (run_type.compareTo("part_b") == 0)
        {
            try 
            {
                long result = PartB.Run(cmd.getOptionValue("f"));
                System.out.printf("Part B result: %d\n", result);
            }
            catch (Exception ex)
            {
                System.err.println("Failed to execute part a");
            }
        }
        else {
            System.out.println("run type Not supported");
            return;
        }
        
        return;
    }
}
