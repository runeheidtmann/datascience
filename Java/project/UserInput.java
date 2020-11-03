import java.util.Scanner;
import java.util.InputMismatchException;

/**
* 
* Userinput class takes care of validation of userinput.
* 
* If a user inputs the wrong type, the class-methods will tell the user and ask for the right type.
* 
*/
public class UserInput{
    
    private static Scanner reader;

    public UserInput(){
        reader = new Scanner(System.in);
    }

    
    /**
     * Takes a string input from user and returns it
     */
    public String inputString(){

        return reader.next(); 
    }

     /**
     * Takes an int input from user and returns it.
     * If user input not valid, the method will prompt user for right input.
     */
    public int inputInt(){
        int userIn = 0;
        boolean valid = false;
        do{
            try {
                userIn = reader.nextInt();
                valid = true;
            }
            catch (InputMismatchException a) {
                System.out.print("You have to type in a whole number: ");
                 reader.nextLine();
            }
        } while (!valid);

        return userIn; 
    }

    /**
     * Takes a double input from user and returns it.
     * If user input not valid, the method will prompt user for right input.
     */
    public double inputDouble(){
        double userIn = 0;
        boolean valid = false;
        while (!valid){
            try {
                userIn = reader.nextDouble();
                valid = true;
            }
            catch (InputMismatchException a) {
                System.out.print("You have to type in a number: ");
                reader.nextLine();
            }
        } 

        return userIn; 
    }
}