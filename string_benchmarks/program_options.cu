#include <iostream>
#include <boost/program_options.hpp>
namespace po = boost::program_options;

#include "program_options.hpp"

/// Initializes command line arguments and their descriptions
po::options_description init_args()
{
    po::options_description desc("Allowed options");
    desc.add_options()
	("help,h", "Produce help")
        ("data_file,d", po::value<std::string>()->default_value("data.in"), "Set filename for data")
        ("pattern_file,p", po::value<std::string>()->default_value("pattern.in"), "Set filename for pattern")
	("pattern_count,c", po::value<int>()->default_value(1), "Set pattern count")
	("pattern_size,a", po::value<int>()->default_value(100), "Set single pattern size in bytes")
	("data_size,s", po::value<int>()->default_value(1048576), "Set data size in bytes");

    return desc;
}

/* Parses command line arguments for benchmark
 * @returns Parsed arguments
 */
Arguments read_arguments(int argc, char** argv)
{
    auto args_desc = init_args();

    Arguments result;
    po::variables_map vm;
    try
    {
        po::store(po::command_line_parser(argc, argv).options(args_desc).style(po::command_line_style::unix_style).run(), vm);
        po::notify(vm);

        result.data_file = vm["data_file"].as<std::string>();
        result.pattern_file = vm["pattern_file"].as<std::string>();
	result.data_length = vm["data_size"].as<int>();
	result.pattern_length = vm["pattern_size"].as<int>();
	result.pattern_count = vm["pattern_count"].as<int>();
        
    }
    catch(po::error &e)
    {
        std::cerr << "Error occured during argument parsing: " << e.what() << std::endl;
        return Arguments();
    }

    return result;
}

