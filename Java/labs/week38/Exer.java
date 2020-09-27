public class Exer{

    public static void main(String[] args){
        System.out.println(downToOne(7));
    }

    public static void linesOfStars(int n){
        int i = 0;
        String stars = "";
        while(i < n){
            stars = stars+"*";
            i = i+1;
            System.out.println(stars);
        }
    }
    public static void stars(int n){
        int i = 0;
        while(i<n){
            System.out.print("*");
            i = i + 1;
        }
    }
    public static void whiteSpace(int n){
        int i = 0;
        while(i<n){
            System.out.print(" ");
            i = i + 1;
        }
    }
    public static void triangle(int n){
        int white = n-1;
        int star = 1;
        int i = 0;
        while(i<n){
            whiteSpace(white);
            stars(star);
            System.out.println("");
            star = star + 2;
            white = white - 1; 
            i = i +1;
        }        
    }
    public static int downToOne(int n){

        int num = n;
        int count = 0;

        while(num != 1){
            
            if(num % 2 == 0)
                num = num / 2;
            else{
                num = num * 3 + 1;
            }
            count = count + 1;
            System.out.println(num);
        }

        return count;
    
    }

}