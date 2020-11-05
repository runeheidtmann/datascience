public class Photoshop{
    public static void main(String[] args){

        Image img = new Image("money.jpg");
        ImageUtils util = new ImageUtils();

        int numOfCols = img.width();
        int numOfRows = img.height();
        
        int red = img.red(0,0);
        int green = img.green(0,0);
        int blue = img.blue(0,0);
 
        Image newimg = util.rotateRight(img);
         Image dickimg = util.rotateRight(newimg);

       
        dickimg.display();
           
    }
}
