public class ImageUtils{


    public Image flipHorizontal(Image img){
        int numOfCols = img.width()-1;
        int numOfRows = img.height()-1;

        Image newimg = new Image(img.width(),img.height());

        int col = 0;
        int row = 0;
        int count = 1;
        while(row <= numOfRows){
            System.out.println("row"+row);
            while(col <= numOfCols){
                int red = img.red(numOfCols-col,row);
                int green = img.green(numOfCols-col,row);
                int blue = img.blue(numOfCols-col,row);
                
                newimg.setPixel(col,row,red,green,blue); 
                col = col + 1;
            }
            col = 0;
            row = row + 1;
        }

        return newimg;
    }
    public Image flipVertical(Image img){
        int numOfCols = img.width();
        int numOfRows = img.height();

        Image newimg = new Image(img.width(),img.height());

        int col = 0;
        int row = 0;
        int count = 1;
        while(col <= numOfCols-1){
            while(row <= numOfRows-1){
                int red = img.red(col,numOfRows-row);
                int green = img.green(col,numOfRows-row);
                int blue = img.blue(col,numOfRows-row);
                
                newimg.setPixel(col,row,red,green,blue); 
                row = row + 1;
            }
            row = 0;
            col = col + 1;
        }

        return newimg;
    }
    public Image rotateRight(Image img){

        Image newImg = new Image(img.height(),img.width());

        int col = 0;
        int row = 0;

        while(col <= img.width()-1){
            
            while(row <= img.height()-1){
                int red = img.red(col,img.height()-row-1);
                int green = img.green(col,img.height()-row-1);
                int blue = img.blue(col,img.height()-row-1);
                newImg.setPixel(row,col,red,green,blue);
                row = row + 1;
            }
            row = 0;
            col = col + 1;
        }

        return newImg;
    }
}