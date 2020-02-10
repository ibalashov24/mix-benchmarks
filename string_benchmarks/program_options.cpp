#include <iostream>
#include <boost/program_options.hpp>
namespace po = boost::program_options;

#include "program_options.hpp"

/// Initializes command line arguments and their descriptions
po::options_description init_args()
{
    po::options_description desc("Allowed options");
    desc.add_options()
        ("data_file,d", po::value<std::string>()->default_value("data.in"), "Set filename for data")
        ("pattern_file,p", po::value<std::string>()->default_value("pattern.in"), "Set filename for pattern")
        ("benchmark_type,b", po::value<std::string>(), "Set benchmark type");

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
        po::store(po::parse_command_line(argc, argv, args_desc), vm);

        result.data_file = vm["data_file"].as<std::string>();
        result.pattern_file = vm["pattern_file"].as<std::string>();
        if (vm.count("benchmark_type"))
        {
            result.benchmark_type = vm["benchmark_type"].as<std::string>();
        }
        else
        {
            std::cout << "You must specify benchmark name!" << std::endl;
        }

        po::notify(vm);
    }
    catch(po::error &e)
    {
        std::cerr << "Error occured during argument parsing: " << e.what() << std::endl;
        return Arguments();
    }

    return result;
}


