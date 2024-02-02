
class PointGroupStatus:
    def __init__(self):
        self.reset()

    def reset(self):
        self.out_of_frame = False
        self.jumped = False

    def point_went_out_of_frame(self):
        self.out_of_frame = True

    def did_point_go_out_of_frame(self) -> bool:
        return self.out_of_frame

    def point_jumped(self):
        self.jumped = True

    def did_point_jump(self) -> bool:
        return self.jumped
