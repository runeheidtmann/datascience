import java.util.Scanner;
import java.io.File;
import java.io.FileNotFoundException;

public class RunSimulation {

    private static UserInput reader = new UserInput();
    private static double sugarProbability;
    private static int avgSugar;
    private static int carriedSugar;
    private static int droppedPheromones;
    private static int numberOfColonies;
    private static int numberOfAntsInColony;
    private static Colony[] homes;
    private static Ant[] ants;
    private static Graph graph;
    private static boolean isGrid;
    private static int width, height;
    private static String filename;
    private static File file;
    private static boolean fromFile;
    private static boolean wantsGraphical;
    private static int frequencyOfReports;
    private static int simulationTime;
    private static Simulator simulator;
    private static Visualizer visualizer;


    public static void main(String[] args) { 
        printStartMessage();

        fromFile = chooseFileOrGrid();

        if (fromFile) {
            try {
                readFile();
            } catch (FileNotFoundException e) {
                System.out.print("Unknown error");
            }
            isGrid = false;   
        } else {
            promptForGridParameters();
            isGrid = width >= 3 & height >= 3;
        }

        promptForParameters();

        initializeColonies();

        initalizeAnts();

        initializeGraph();

        promptForVisualizationParameters();
        
        simulator = new Simulator(graph, ants, carriedSugar, droppedPheromones);
        
        visualizer = new Visualizer(graph, isGrid, ants[0].current(), ants);

        runSimulation();
    }

    /*
     *  Print a nice start message
     */
    private static void printStartMessage() {    
        System.out.println("               ,                                            ");
        System.out.println("      _,-'\\   /|   .    .    /`.                           ");
        System.out.println("  _,-'     \\_/_|_  |\\   |`. /   `._,--===--.__            ");
        System.out.println(" ^       _/\"/  \" \\ : \\__|_ /.   ,'    :.  :. .`-._      ");
        System.out.println("        // ^   /7 t'\"\"    \"`-._/ ,'\\   :   :  :  .`.    ");
        System.out.println("        Y      L/ )\\         ]],'   \\  :   :  :   : `.    ");
        System.out.println("        |        /  `.n_n_n,','\\_    \\ ;   ;  ;   ;  _>   ");
        System.out.println("        |__    ,'     |  \\`-'    `-.__\\_______.==---'     ");
        System.out.println("       //  `\"\"\\\\      |   \\            \\              ");
        System.out.println("       \\|     |/      /    \\            \\                ");
        System.out.println("                     /     |             `.                 ");
        System.out.println("                    /      |               ^                ");
        System.out.println("                   ^       |                                ");
        System.out.println("              _         _  ^              _       _         ");
        System.out.println("   __ _ _ __ | |_   ___(_)_ __ ___  _   _| | __ _| |_ ___  _ __     ");
        System.out.println("  / _` | '_ \\| __| / __| | '_ ` _ \\| | | | |/ _` | __/ _ \\| '__| ");
        System.out.println(" | (_| | | | | |_  \\__ \\ | | | | | | |_| | | (_| | || (_) | |     ");
        System.out.println("  \\__,_|_| |_|\\__| |___/_|_| |_| |_|\\__,_|_|\\__,_|\\__\\___/|_| ");
        System.out.println("                                                                    ");       
    }


    /*
     *  Ask the user to choose between loading a file or generating a grid graph
     */
    private static boolean chooseFileOrGrid() {
        System.out.println("----------------------------------------------------------------");
        System.out.println("               M A I N   M E N U                 ");
        System.out.println("            ------------------------------------");
        System.out.println("            1. Load graph from file              ");
        System.out.println("            2. Create new rectangular grid graph ");
        System.out.println("----------------------------------------------------------------");
        System.out.println();
        
        String choice = "";
        do {
            System.out.print("Enter your choice: ");
            choice = reader.inputString();
            if (!choice.equals("1") && !choice.equals("2")) {
                System.out.println("Please enter 1 or 2.");
            }
        } while (!choice.equals("1") && !choice.equals("2"));
        
        // the boolean fromFile is set to true, if choice equals "1"
        fromFile = choice.equals("1");

        return fromFile;
    }

    /*
     *  Prompts the user for a filename until the filename corresponds to a
     *  file containing a valid graph.
     *  Then reads the number of colonies specified in the file.
     */
    private static void readFile() throws FileNotFoundException {
        boolean fileLoaded = false;
        while (!fileLoaded) {
            System.out.print("Enter the filename: ");
            filename = reader.inputString();
            try {
                File file = new File(filename);
                fileLoaded = true;
                if (containsDuplicateEdges(file)) {
                    System.out.println("Error: Graph contains duplicate edges.");
                    fileLoaded = false;
                } else {
                    // read the number of colonies specified in the file
                    getInfoFromFile(file);
                    System.out.println("\n~~~~~~~~~~~~~~ Graph loaded successfully ~~~~~~~~~~~~~~~~~~~~~~\n");
                }
            } catch (FileNotFoundException e) {
                System.out.println("Error: Unknown file");
                fileLoaded = false;
            }
        }
    }

    /*
     *  Returns true if the file contain no duplicate edges.
     *
     *  Assumes that the file structure is as follows: one line containing the 
     *  number of nodes (including colonies); one line containing the nodes 
     *  corresponding to colonies; and a variable number of lines describing 
     *  edges (one per line). The nodes are assumed to be numbered sequentially 
     *  from 1, and each edge is described by a pair of numbers (the source and 
     *  target nodes of the edge).
     */
    private static boolean containsDuplicateEdges(File file) throws FileNotFoundException {
        // read from the file
        Scanner fileReader = new Scanner(file);
        fileReader.nextLine();  // skip the first two lines
        fileReader.nextLine();

        // append all edges in file to a long string
        String allEdges = "";
        while (fileReader.hasNextLine()) {
            allEdges = allEdges + fileReader.nextLine().trim() + ",";
        }

        // create String array with all the edges
        String[] edges = allEdges.split(",");

        // check the array for duplicates
        boolean duplicate = false;
        int i = 0;
        while (!duplicate && i < edges.length - 1) {
            int j = i + 1;
            while (!duplicate && j < edges.length) {
                if (edges[i].equals(edges[j])) {
                    duplicate = true;
                }
                j = j + 1;
            }
            i = i + 1;
        }

        return duplicate;
    }


    /*
     *  Reads the number of colonies specified in the file.
     *
     *  Assumes that the file structure is as follows: one line containing the 
     *  number of nodes (including colonies); one line containing the nodes 
     *  corresponding to colonies; and a variable number of lines describing 
     *  edges (one per line). The nodes are assumed to be numbered sequentially 
     *  from 1, and each edge is described by a pair of numbers (the source and 
     *  target nodes of the edge).
     */
    private static void getInfoFromFile(File file) throws FileNotFoundException {
        Scanner fileReader = new Scanner(file);
        fileReader.nextLine();  // skip the first line
        String lineWithListOfColonies = fileReader.nextLine().trim();
        numberOfColonies = lineWithListOfColonies.split(" ").length;
    }



    private static void promptForGridParameters() {
        System.out.print("Enter the width of the grid: ");
        width = reader.inputInt();

        System.out.print("Enter the height of the grid: ");
        height = reader.inputInt();

        System.out.println("\n~~~~~~~~~~~~~~ Grid created successfully ~~~~~~~~~~~~~~~~~~~~~~\n");
    }



    private static void promptForParameters() {
        System.out.println("---------------------------------------------------------------");
        System.out.println("               S I M U L A T I O N  P A R A M E T E R S        ");
        System.out.println("---------------------------------------------------------------");
        
        System.out.println("\n<1/7>\nThe number of ant colonies");
        if (fromFile) {
            System.out.println("Already set to " + numberOfColonies + " (from file)");
        } else {
            System.out.print("Enter number: ");
            numberOfColonies = reader.inputInt();
        }

        System.out.println("\n<2/7>\nThe number of ants initially in each colony.");
        System.out.print("Enter number: ");
        numberOfAntsInColony = reader.inputInt();

        
        System.out.println("\n<3/7>\nThe probability that a node will start as a node containing sugar.");
        do{
        System.out.print("Enter probability between 0 and 1: ");
            sugarProbability = reader.inputDouble();
        } while(sugarProbability >= 1);
        

        System.out.println("\n<4/7>\nThe average amount of sugar in such a node.");
        System.out.print("Enter amount: ");
        avgSugar = reader.inputInt();

        System.out.println("\n<5/7>\nThe number of units of sugar that an ant can carry.");
        System.out.print("Enter number: ");
        carriedSugar = reader.inputInt();

        System.out.println("\n<6/7>\nThe number of units of pheromones dropped by an ant passing a node.");
        System.out.print("Enter number of units: ");
        droppedPheromones = reader.inputInt();

        System.out.println("\n<7/7>\nThe total simulation time.");
        System.out.print("Enter number of ticks: ");
        simulationTime = reader.inputInt();

        System.out.println("\n~~~~~~~~~~~~~~ All simulation parameters set ~~~~~~~~~~~~~~~~~~~\n");
    }


    private static void initializeColonies() {
        homes = new Colony[numberOfColonies];
        int i = 0;
        while (i < homes.length) {
            homes[i] = new Colony();
            i = i + 1;
        }
    }

    private static void initalizeAnts() {
        ants = new Ant[numberOfColonies * numberOfAntsInColony];      
        int index = 0;
        int i;
        for (Colony home : homes){
            i = 0;
            while (i < numberOfAntsInColony) {
                ants[index] = new Ant(home);
                i = i + 1;
                index = index + 1;
            }
        }
    }


    private static void initializeGraph() {
        if (fromFile) {
            graph = new Graph(filename, homes, sugarProbability, avgSugar);
            
        } else {
           graph = new Graph(width, height, homes, sugarProbability, avgSugar);
        }
    }


    private static void promptForVisualizationParameters() {
        String answer = "";
        do {
            System.out.println("----------------------------------------------------------------");
            System.out.println("               C H O O S E  V I E W I N G  M O D E              ");
            System.out.println("            -----------------------------------------           ");
            System.out.println("            1. Textural summary                                 ");
            System.out.println("            2. Graphical representation                         ");
            System.out.println("----------------------------------------------------------------");
            System.out.print("\nEnter choice: ");
            answer = reader.inputString();
            if (answer.equals("1")) {
                wantsGraphical = false;
                System.out.print("Enter number of ticks between reports: ");
                frequencyOfReports = reader.inputInt();
            } else if (answer.equals("2")) {
                wantsGraphical = true;
            } else {
                System.out.println("Please enter 1 or 2");
            }
        } while (!answer.equals("1") && !answer.equals("2"));
    }


    private static void runSimulation() {     
        System.out.println();
        System.out.println("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~");
        System.out.println("               R U N N I N G  S I M U L A T I O N              ");
        System.out.println("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~");
        System.out.println();
        
        if (wantsGraphical) {
            visualizer.display();
            for (int t = 0; t <= simulationTime; t = t + 1) {
                simulator.tick();
                visualizer.update();
            }
        } else {
            for (int t = 0; t <= simulationTime; t = t + 1) {
                simulator.tick();
                if (t % frequencyOfReports == 0 || t == simulationTime) {
                    System.out.println("\n----------------------------------------------------------------");
                    System.out.println("Step: " + t + " out of " + simulationTime);
                    System.out.println("----------------------------------------------------------------");
                    visualizer.printStatus();
                }
            }
        }
    }

}