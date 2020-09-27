public class Arrays{

    public static double sum(double[] v){
        
        double sum = 0;
        //For each double d in double-array v
        for(double d : v){
            sum = sum + d;
        }
        return sum;
    }

    public static int zeros(int[] v){
        
        int zeroCounter = 0;

        for( int number : v){
            if(number == 0)
                zeroCounter = zeroCounter + 1;
        }

        return zeroCounter;
    }

    public static int count(int[] v, int n){
        int numberCounter = 0;
        for(int number : v){
            if(number == n)
                numberCounter = numberCounter + 1;
        }
        return numberCounter;
    }

    public static int smallerThan(int[] v, int n){
        
        int numberCounter = 0;
        for(int number : v){
            if(number < n)
                numberCounter = numberCounter + 1;
        }
        return numberCounter;
    }
    
    public static boolean member(int[] v, int n){

        for(int number : v){
            if(number == n)
                return true;
        }
        return false;
    }

    public static boolean twoZeros(int[] v){

        boolean lastNumberSeenWasZero = false;
        
        for(int number : v){

            if(number == 0){
                
                if(lastNumberSeenWasZero){
                    return true;
                }

                lastNumberSeenWasZero = true;
                continue;
            }

            lastNumberSeenWasZero = false;
        }
        
        return false;
    }

    public static String toString(int[] v){
        
        String arrayAsString = "[";
        boolean firstNumber = true;

        for(int number : v){
            if(!firstNumber){
                arrayAsString = arrayAsString + ", ";
            }
            arrayAsString = arrayAsString + Integer.toString(number);
            firstNumber = false;
        }
        arrayAsString = arrayAsString + "]";
        return arrayAsString;
    }

    public static int[] squares(int n){

        int[] arrayOfSquares = new int[n];
        
        int numToBeSquared = 1;
        while(numToBeSquared <= n){
           arrayOfSquares[numToBeSquared-1] = numToBeSquared*numToBeSquared;
           numToBeSquared = numToBeSquared + 1;
        }

        return arrayOfSquares;
 
    }
    public static int[] decreasingSquares(int n){

        int[] arrayOfSquares = new int[n];
        
        int numToBeSquared = n;
        int i = 0;
        while(numToBeSquared >= 1){
           arrayOfSquares[i] = numToBeSquared*numToBeSquared;
           numToBeSquared = numToBeSquared - 1;
           i = i + 1;
        }

        return arrayOfSquares;
    }
    public static int countDivisors(int n){
         int divisorCount = 0;
         
        for(int i = 1; i <= n; i++){
            if(n%i == 0)
                divisorCount++;
        }
        return divisorCount;    
    }

    public static int[] divisors(int n){
        
        //get number of divisors in anotther function.
        int[] divisors = new int[countDivisors(n)];

        int divIndex = 0;
        int i = 1;
        while(i<=n){
            if(n%i == 0){
                divisors[divIndex] = i;
                divIndex = divIndex + 1;
            }
            i = i + 1;
        }
        return divisors;   
    }
    public static double max(double[] v){
        double max = v[0];
        
        for(double d : v){
            if(d > max)
                max = d;
        }
        return max;
        
    }

    public static boolean subset(int[] v, int[] w){
        
    }


    public static void main(String[] args){
    
        double[] doubles = {8.2,2.5,0,4.3,4,15.1,160,1,151};
        int n = 16;
        System.out.print(max(doubles));
    }
}