
#ターミナルで入力する場合

import numpy as np
import os

#ターミナル入力=1
cmd_input = 0

class HumanAgent:
    def __init__(self,aid,actions=None):
        self.actions = actions
        self.aid = aid
    
    def act(self):
        #キー入力
        if cmd_input == 1:
            while(True):
                print("w:↑ a:← s:↓ d:→ x:stay")
                input_key = input('>>>')

                if input_key == "w":
                    return 2
                elif input_key == "a":
                    return 0
                elif input_key == "s":
                    return 3
                elif input_key == "d":
                    return 1
                elif input_key == "x":
                    return 4
                else:
                    print("その操作はありません")
        else:
            pass

    
    def react(self):
        print("その方向には移動できませんでした")
        if cmd_input == 1:
            while(True):
                print("w:↑ a:← s:↓ d:→ x:stay")
                input_key = input('>>>')

                if input_key == "w":
                    return 2
                elif input_key == "a":
                    return 0
                elif input_key == "s":
                    return 3
                elif input_key == "d":
                    return 1
                elif input_key == "x":
                    return 4
                else:
                    print("その操作はありません")
        else:
            pass
    
    def observe(self,next_state,reward=None,reward_o=None,opponent_action=None,is_learn=False):
        pass