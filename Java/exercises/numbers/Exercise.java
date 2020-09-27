import java.lang.Math; 
import java.util.*;
import java.util.Timer;
import java.util.LinkedList;



public class Exercise{ 
      
    public static int count = 0;
    public static void main(String[] args){

    // The exercise-methods you want to run:
     //first make a list containing all the uneven numbers up to n (since the even numbers are not prime):
        // int number = 1000000;
         ArrayList<Integer> numbers = new ArrayList<Integer>();
        // numbers.add(2);
        for(int i = 1; i<1000000; i++){
            numbers.add(i);
        }
     
        sieveEratosthenes(numbers);
        System.out.println(numbers.size());
        
    }

    // Exer 2.1: 
    // Just adding 7 to i and printing it out

    public static void printMultiplesOfSevenA(){
        for(int i = 7; i<500; i=i+7){
            System.out.println(i);
        }
    }

    // Exer 2.2:
    // running through all the numbers and print it out if it divides with 7
    // Note: This one is computationally harder than the A-version of printMultiples.

    public static void printMultiplesOfSevenB(int n){

        for(int i = 7; i<n ; i++){
            if(i%7 == 0)
                System.out.println(i);
        }
    }

    //Exer 2.3
    //take in k and and and start at that number. Then print it out, add k and print again. Do this while i is less than n.

    public static void printMultiplesOf(int k, int n){

        for(int i = k; i<n; i=i+k){
            System.out.println(i);
        }
    }

    // Exer 2.4
    // We calculate som of all numbers lower than n, by running through them all and add them to the sum.
    public static void sumUpTo(int n){
        int sum = 0;
        for(int i = 0; i<n;i++)
            sum = sum+i;
        
        System.out.println(sum);
    }
    
    /* Exer 2.5
    
            n | 1 | 2 | 3 | 4  | 5  | 6  | 7  | 8  | 9  |  10
            -------------------------------------------
        sum(n)| 1 | 3 | 6 | 10 | 15 | 21 | 28 | 36 | 45 | 55  
    
    So, we are after the smallest n, where sum(n) is bigger than k.
    Example:
        k = 14 than the smallest n where sum(n) is bigger than k is n = 5

    */
    public static void sumBeyond(int k){

        int sum = 0;
        int count = 0;
        while(sum < k){
            count++;
            sum = sum+count;
        }
    
        System.out.println(count);
    
    }

    /* Exer 2.6
       Sum of numbers between m and k. 
    */
    public static void sumBetween(int m, int k){
       
        int sum = 0;

        for(int i = m; i<=k; i++){
            sum = sum + i;
        }
        System.out.println(sum);
    }

    /* Exer 2.7
       Sum of all even numbers lower than n.
       To save computation we add every other number together.
     */

    public static void sumEven(int n){

        int sum = 0;

        for(int i = 0; i<n ; i = i+2){
            sum = sum + i;
        }
        System.out.println(sum);
    }

    /* Exer 2.8
        It didnt take long for Luíz, to blow our minds with a recursion exercise.    
        Now, you surely can solve this with a forloop. But the "mathyness" of the factorial function
        kinda calls for this special kind of function in programming called recursion. All you need to know, is that its a function that calls itself.
        Here we go:

        factorial(5) is defined by 5*4*3*2*1
        We see that factorial(4) is 4*3*2*1
        Therefor factorial(5) = 5*fact(4) = 5*4*fact(3) = 5*4*3*fact(2) = 5*4*3*2*1
        
        In other words:
    */  
    public static int factorial(int n){      
        if(n == 1)
            return 1;

        return n*factorial(n-1); 
    }

    /* Exer 2.9 
       Double factorial is written n!! and defined by the product of all the integers from 1 up to n that have the same parity (odd or even) as n
       Programmatically we kan solve this in the excact same way as n!, but instead do (n-2):

       Note that the base cases now are 2 or 1 instead of just 1. 
       Base case for odd numbers n -> 1
       Base case for even numbers n -> 2 
    */
    public static int doubleFactorial(int n){
        
        if(n == 2)
            return 2;
        
        else if(n == 1)
            return 1;

        return n*doubleFactorial(n-2);
    }

    /* Exer 2.10
       Fibonacci is defined as fib(n) = fib(n-1) +fib(n-2) where fib(0) = 0 and fib(1) = fib(2) = 1
       You see, how the defintion of fn has a base case f(0) = 0 and another base case f(1) = 1
       Again, well solve this with recursion, because of the nature of the fibonacci math.
       What the code does:
       fib(6) = fib(5)+fib(4) 
              = fib(4)+fib(3)+fib(3)+fib(2) 
              = fib(3)+fib(2)+fib(2)+fib(1)+fib(2)+fib(1)+fib(1)+fib(0) 
              = fib(2)+fib(1)+fib(2)+fib(2)+fib(1)+fib(2)+fib(1)+fib(1)+fib(0) 
              = 1+1+1+1+1+1+1+1+0 
              = 8
    */
    public static int fibonacci(int n){
        if(n==1)
            return 1;
        else if(n==0)
            return 0;
        
        return fibonacci(n-1)+fibonacci(n-2);
    }

    /* Exer 2.11
       We want the base 2 logarithm om n, also called the binarylogarithm.
       It is defined log2(n) = log10(n) / log10(2)

       We will just use Javas build in Math.log function to get the base 10 logs.
       In order to use this function, we have to import the math class: java.lang.Math; 

    */
    public static double binaryLog(int n){

        return Math.log(n)/Math.log(2);
    }

    /* Exer 2.12
       Again, we are facing a pretty mathy question: Find the number of divisors of any n.
       We could do this by brute force. And we will in example divisorA.
       But we can also do it in a computationally much more cheaper way.
    */
    
    public static int divisorA(int n){
        //Brute force - check all numbers:
        int divisorCount = 0;
        
        for(int i = 1; i <= n; i++){
            if(n%i == 0)
                divisorCount++;
        }
        return divisorCount;    
    }

    /* Exer 2.13
        This one we wil split up in to methods. One for finding all divisors of n. One for calculating if n is a perfect number.
        Example of a perfect number: 6 = 3+2+1 aka. The last number is equal to the som of all the other divisors.
       
    */
    public static boolean isPerfectNumber(int n){

        List<Integer> divisors = findDivisors(n);
        int sumOfDivisors = 0;
        int lastNumberInList = divisors.size()-1;
        
        for(int i = 0; i < divisors.size()-1; i++){
            sumOfDivisors = sumOfDivisors+divisors.get(i);
        }
        
        if(sumOfDivisors == divisors.get(lastNumberInList)){
            return true;
        }

        return false;
    }
    public static List<Integer> findDivisors(int n){
        
        List<Integer> divisors = new ArrayList<Integer>();  
        
        for(int i = 1; i <= n; i++){
            if(n%i == 0)
                divisors.add(i);
            }
        return divisors;
    }

    /*  Exer 2.14
        Print the perfect numbers smaller than n
     */
    public static void printAllPerfectNumbersSmallerThan(int n){

        for(int i = 1; i<n; i++){
            if(isPerfectNumber(i))
                System.out.println(i);
        }
    }

    /* Exer 2.15
       Check if n is a prime.
     */
     public static boolean isPrime(int n){

         if(n<2)
            return false;

         if(n == 2)
            return true;
         
         if(n%2 == 0){
             return false;
         }
         for(int i = 3; i < Math.sqrt(n)+1; i++){
             if(n%i == 0)
                {
                    
                    return false;
                }
                
         }
         return true;
     }

    /*  Exer 2.16
        Find count of primes under int n.
        We run through all an check with isPrime. If true count++
    */
    public static int countPrimes(int n){

        int primeCount = 0;
        
        for(int i = 1; i<n; i++){
            if(isPrime(i))
                primeCount++;
        }

        return primeCount;
    }
    /* Exer 2.16 extra
        We will try to implement the ancient algorithm, Sieve of Eratosthenes, for finding all primes up to a given limit.
        It starts by eliminating all the numbers that are divisible by the first prime -> 2. There after elimination all numbers divisible by the second prime 3 and than 5 and so forth.
    */
    public static List<Integer> sieveEratosthenes(ArrayList<Integer> numbers){

        int baseNum = 0;
        int calc = 0;
        while(baseNum < Math.sqrt(numbers.size())){            
            for(int i = 1; i < numbers.size()-baseNum; i++){
                if(numbers.get(baseNum+i) % numbers.get(baseNum) == 0)
                {
                    
                    // System.out.println("baseNum = " + baseNum + ", removed: "+(numbers.get(baseNum+i)));
                    numbers.remove(baseNum+i);
                    count++;
                }
            }
            // System.out.println("size :" + numbers.size());                        
            baseNum++;
        }
        
        System.out.println("Calcs: "+count);
        return numbers;
    }
    /*public static void realSieve()  
    {  
        int MAX_SIZE = 1000000; 
        int totalt;  
         // To store all prime numbers 
        static ArrayList<Integer> primes = 
        new ArrayList<Integer>();
        
        boolean [] IsPrime = new boolean[MAX_SIZE];  
          
        for(int i = 0; i < MAX_SIZE; i++) 
            IsPrime[i] = true; 
          
        for (int p = 2; p * p < MAX_SIZE; p++)  
        {  
            // If IsPrime[p] is not changed,  
            // then it is a prime  
            if (IsPrime[p] == true)  
            {  
                // Update all multiples of p greater than or  
                // equal to the square of it  
                // numbers which are multiple of p and are  
                // less than p^2 are already been marked.  
                for (int i = p * p; i < MAX_SIZE; i += p){  
                    totalt++;
                    IsPrime[i] = false;  
                    }
            }  
        }  
      
        // Store all prime numbers  
        for (int p = 2; p < MAX_SIZE; p++)  
        if (IsPrime[p] == true)  
                primes.add(p);
        System.out.println(primes);
    }  
*/

}
