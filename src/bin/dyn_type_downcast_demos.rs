fn dyn_type_downcast_Demo() {
    trait Animal {
        fn make_sound(&self);
    }

    #[derive(Debug)]
    struct  Dog;
    impl Animal for Dog {
        fn make_sound(&self) {
            println!("Woof!");
        }
    }

    #[derive(Debug)]
    struct Cat;
    impl Animal for Cat {
        fn make_sound(&self) {
            println!("Meow!");
        }
    }

    #[derive(Debug)]
    struct Duck;
    impl Animal for Duck {
        fn make_sound(&self) {
            println!("Quack!");
        }
    }

    fn make_sound(animal: &dyn Animal) {
        animal.make_sound();
    }

    use std::any::Any;

    let animal: Box<dyn Any> = Box::new(Dog);
    if let Ok(dog) = animal.downcast::<Dog>() {
        dog.make_sound();
    } else {
        println!("Not a dog");
    }
}

fn main() {
    dyn_type_downcast_Demo();
}