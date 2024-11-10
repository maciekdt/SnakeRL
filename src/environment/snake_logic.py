import random
from enum import Enum

class Direction(Enum):
    UP = (0,1)
    DOWN = (0,-1)
    LEFT = (-1,0)
    RIGHT = (1,0)
    
 
class SnakeLogic:
    BOARD_SIZE = 15
    
    def __init__(self):
        self.snake_list = [(
            int(self.BOARD_SIZE/2),
            int(self.BOARD_SIZE/2)
        )]
        
        self.all_positions= set()
        for x in range(self.BOARD_SIZE):
            for y in range(self.BOARD_SIZE):
                self.all_positions.add((x,y))
                
        self.apple_position = self._get_new_random_apple_position()
        self.current_direction: Direction = None
        self.game_over = False
                
        
    def _get_new_random_apple_position(self):
        free_positions = self.all_positions - set(self.snake_list)
        return random.choice(list(free_positions))
    
    def _transform_point(self, point):
        return (point[0] + self.current_direction.value[0], point[1] + self.current_direction.value[1])
    
    def _is_correct_action(self, action: Direction):
        if(self.current_direction == None): return True 
        if(self.current_direction == Direction.UP and action == Direction.DOWN): return False
        if(self.current_direction == Direction.DOWN and action == Direction.UP): return False
        if(self.current_direction == Direction.LEFT and action == Direction.RIGHT): return False
        if(self.current_direction == Direction.RIGHT and action == Direction.LEFT): return False
        return True
        
    def step(self, action: Direction):
        if(self.game_over):
            raise RuntimeError("Game has already ended")
        if self._is_correct_action(action):   
            self.current_direction = action
        new_snake_head = self._transform_point(self.snake_list[-1])
        got_apple = False
        if new_snake_head in set(self.snake_list) or new_snake_head not in self.all_positions: 
            self.game_over = True
        elif new_snake_head == self.apple_position:
            self.snake_list.append(new_snake_head)
            self.apple_position = self._get_new_random_apple_position()
            got_apple = True
        else:
            self.snake_list.append(new_snake_head)
            self.snake_list.pop(0)
            
        return self.snake_list, self.apple_position, self.game_over, got_apple