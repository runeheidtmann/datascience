public class Main{
    public static void main(String[] args){
        
        UserInput input = new UserInput();

        double doubles = input.inputDouble();
        System.out.print(doubles);
        doubles = input.inputDouble();
        System.out.print(doubles);
    }
}