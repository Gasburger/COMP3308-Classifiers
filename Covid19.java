import java.util.Collections;
import java.util.LinkedList;
import java.util.Scanner;

class Covid19 {
    String countryName;
    int cases;

    public Covid19(String countryName, int cases){
               this.countryName = countryName;
               this.cases = cases;
    }

    public static void main(String[] args){

      Scanner scan = new Scanner(System.in);
      System.out.println("Enter data:");
      

      LinkedList<Covid19> caseList = new LinkedList<Covid19>();
      caseList.add(new Covid19(country, cases));

      Collections.sort(caseList, (Covid19 c1, Covid19 c2) -> c1.cases.compareTo(c2.cases), Collections.reverseOrder());
    }

}