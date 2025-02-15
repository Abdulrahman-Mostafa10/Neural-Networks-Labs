import random

class NegativeNumberError(Exception):
    pass

class InvalidInputError(Exception):
    pass

class FloatNumberError(Exception):
    pass

def guessing_game(max: int, attempts: int) -> tuple[bool, list[int], int]:
    secret_number:int = random.randint(1, max)
    
    guesses:list = []
    
    for attempt in range(attempts):
        try:
            guess:str = input(f"Attempt {attempt + 1}/{attempts}: Guess a number between 1 and {max}: ")
            
            if '.' in guess:
                raise FloatNumberError("Floats are not allowed. Please enter an integer")
            
            if not guess.lstrip('-').isdigit():
                raise InvalidInputError("Invalid input. Please enter a positive integer")
            
            guess :int = int(guess)
            
            if guess < 0:
                raise NegativeNumberError("Negative numbers are not allowed")
            
            if guess < 1 or guess > max:
                print(f"Please enter a number between 1 and {max}")
                continue 
            
            guesses.append(guess)
            
            if guess == secret_number:
                print("Congratulations! You guessed the correct number")
                return True, guesses, secret_number 
            elif guess < secret_number:
                print("Too low! Try again")
            else:
                print("Too high! Try again")
        
        except FloatNumberError as e:
            print(e)  
        except InvalidInputError as e:
            print(e)  
        except NegativeNumberError as e:
            print(e)  
        except ValueError:
            print("Invalid input. Please enter a valid integer")
    
    return False, guesses, secret_number
    
        

# TODO 
def play_game()-> None:
    max_value:int = 20
    attempts:int = 5
    
    while True:
        isWon, guesses, chosen_int = guessing_game(max_value,attempts=attempts)

        if isWon:
            assert chosen_int in guesses, "Error: The secret number should be in the guesses list"
            break
        
        else:
            assert chosen_int not in guesses, "Error: The secret number shouldn't be in the guesses list"
            play_again = input("Do you want to play again? (yes/no): ").strip().lower()
            if play_again != "yes":
                break
play_game()