
class PatchPow2Info:
    def __init__(self, idx: int, pow_2: int):
        self.idx = idx
        self.pow_2 = pow_2
    
    def __str__(self) -> str:
        return '%s, pow2: %s' % (self.idx, self.pow_2)

def sort_infos(infos: list[PatchPow2Info]):
    ''' Sort largest to smallest power of 2 '''
    infos.sort(key=lambda x: x.pow_2, reverse=True)

def find_best_infos(pow_2: int, infos: list[PatchPow2Info]) -> list[int]:
    '''
    @param pow_2 Min power of 2 to look for in infos
    @param infos Expecting this to be sorted with sort_infos
    @return best_idxs A list containing the PatchPow2Info.idx values from
           infos.pow_2 that is greater than or equal to the input pow_2
    '''
    best_idxs = [ ]
    last_pow_2 = 0
    for info in infos:
        if info.pow_2 < pow_2 and best_idxs and last_pow_2 != info.pow_2:
            break
        best_idxs.append(info.idx)
        last_pow_2 = info.pow_2
    return best_idxs
