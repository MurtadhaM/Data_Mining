public class Bruh {
    public static void main(String[] args) { 
        Person Meme = new Person();
        Meme.name = "Mansoor";
        Meme.age = 20;


        System.out.println(Meme.toString());
    }
}
class Person{
    public String name;
    public int age;
    public String toString(){
        return "Name:" + name + "\nAge: " + age;
    }
}