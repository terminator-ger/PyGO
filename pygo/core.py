from re import I
import cv2
import pdb
import logging
from pygo.BoardTracker import BoardTracker
from pygo.Keyframes import History

from pygo.classifier import *
from pygo.Motiondetection import *
from pygo.GoBoard import GoBoard
from pygo.utils.plot import Plot
from pygo.utils.debug import Timing
from pygo.Game import Game
from pygo.Webcam import Webcam
from pygo.Signals import *

logging.DEBUG2 = 5
logging.addLevelName(logging.DEBUG2, "DEBUG2")
logging.Logger.debug2 = lambda inst, msg, *args, **kwargs: inst.log(logging.DEBUG2, msg, *args, **kwargs)
logging.debug2 = lambda msg, *args, **kwargs: logging.log(logging.DEBUG2, msg, *args, **kwargs)



class PyGO(Timing):
    def __init__(self):
        Timing.__init__(self)
        self.input_stream = Webcam()

        self.img_cam = self.input_stream.read()
        self.img_overlay = self.img_cam
        self.img_cropped = self.img_cam
        self.img_virtual = self.img_cam

        self.History = History()
        self.Board = GoBoard(self.input_stream.getCalibration())
        self.Plot = Plot()
        self.Game = Game()
        self.PatchClassifier = CircleClassifier(self.Board, 19)
        self.Motiondetection = MotionDetectionMOG2(self.img_cam, classifier=self.PatchClassifier)
        self.BoardTracker = BoardTracker()

        self.kf_thresh = 3

        self.msg = ''
        self.Katrain = None
        self.input_is_frozen = False
    
        Signals.subscribe(GameRun, self.unfreeze)
        Signals.subscribe(GamePause, self.freeze)
        Signals.subscribe(GameNew, self.startNewGame)
        Signals.subscribe(GameNewMove, self._notify_ui_new_move)
        Signals.subscribe(DetectHandicap, self.__analyzeHandicap)
        Signals.subscribe(PreviewNextFrame, self.__force_image_load)

    def freeze(self, *args) -> None:
        self.input_is_frozen = True 

    def unfreeze(self, *args) -> None:
        self.input_is_frozen = False

    def __analyzeHandicap(self, *args) -> None:
        if self.Board.hasEstimate:
            self.img_cam = self.input_stream.read_ignore_lock()
            self.img_cropped = self.Board.extract(self.img_cam)
            val = self.PatchClassifier.predict(self.img_cropped)
            self.Game._check_handicap(val)
        else:
            logging.warning('No Board detected - make sure to place the board \
                             in the cameras center')

    def __force_image_load(self, args) -> None:
        if self.Board.hasEstimate:
            self.img_cam = self.input_stream.read_ignore_lock()
            self.img_cropped = self.Board.extract(self.img_cam)
 

    def startNewGame(self, size=19) -> None:
        #unfreeze to allow reading of new frame

        self.Game.startNewGame(19)
        self.img_cam = self.input_stream.read_ignore_lock()
        if not self.Board.hasEstimate:
            self.Board.calib(self.img_cam)
        
        if self.Board.hasEstimate:
            self.img_cropped = self.Board.extract(self.img_cam)

        # pause the game 
        Signals.emit(GamePause)


    def loop10x(self) -> None:
        for _ in range(10):
            img_cam = self.input_stream.read()
        for _ in range(10):
            self.run_once()
            print(self.input_is_frozen)

    def loopDetect10x(self) -> None:
        for _ in range(10):
            img_cam = self.input_stream.read()
        self.Board.calib(img_cam)
        for _ in range(10):
            self.run_once()


    def loop(self) -> None:
        while (True):
            self.run_once()


    def update_history(self, prediction: GoBoardClassification) -> None:
        state = self.Game.state
        t = self.input_stream.get_time()
        self.History.update(t, prediction, state)


    def _notify_ui_new_move(self, args):
        colour = args[0]
        t = self.input_stream.get_time()
        Signals.emit(UIDrawStoneOnTimeline, colour, t)

    
    def run_once(self) -> None:
            if not self.input_is_frozen:
                self.img_cam = self.input_stream.read()

            if self.Board.hasEstimate:
                self.img_cropped =  self.Board.extract(self.img_cam)

                if self.Motiondetection.hasNoMotion(self.img_cropped) \
                    and not self.Game.isPaused():

                    
                    if self.PatchClassifier.hasWeights:
                        if self.BoardTracker.update(self.Board.track_corners(self.img_cam)):
                            self.Board.calib(self.img_cam)
                            self.img_cropped =  self.Board.extract(self.img_cam)

                        val = self.PatchClassifier.predict(self.img_cropped)

                        self.img_overlay = self.Plot.plot_overlay(val, 
                                                        self.Board.go_board_shifted, 
                                                        self.img_cropped,
                                                        self.Game.manualMoves,
                                                        self.Game.last_x,
                                                        self.Game.last_y,
                                                        self.Board.border_size)
                        self.img_virtual = self.Plot.plot_virt_grid(val, 
                                                        self.Board.grd_overlay, 
                                                        self.Board.grid_img,
                                                        self.Game.manualMoves,
                                                        self.Game.last_x,
                                                        self.Game.last_y)


                        self.Game.updateState(val)
                        self.update_history(val)

                        if self.input_stream.is_video:
                            last_t, kf_count = self.History.how_many_kf_since_last_update()
                            logging.warning("{} Keyframes added without an update to the game state".format(kf_count))
                            if kf_count > 0:
                                pass
                                #self.backtrack(from_=last_t)

                        #if self.Katrain is not None:
                        #    self.Katrain.send(self.msg)
                else:
                    #overlay old state during motion
                    self.img_overlay = self.Plot.plot_overlay(self.Game.state,
                                                                self.Board.go_board_shifted,
                                                                self.img_cropped,
                                                                self.Game.manualMoves,
                                                                self.Game.last_x,
                                                                self.Game.last_y,
                                                                self.Board.border_size)
                    self.img_virtual = self.Plot.plot_virt_grid(self.Game.state, 
                                                        self.Board.grd_overlay, 
                                                        self.Board.grid_img,
                                                        self.Game.manualMoves,
                                                        self.Game.last_x,
                                                        self.Game.last_y)
            else:
                img = self.Board.get_corners_overlay(self.img_cam)
                self.img_overlay = img
                self.img_virtual = img
                self.img_cropped = img
            

    def backtrack(self, from_=None) -> None:
        # halt the stream
        Signals.emit(GamePause)
        to_ = self.input_stream.get_time()

        self.input_stream.set_pos((from_,))
        last_kf = from_
        logging.debug("Backtracking from {} to {}".format(from_, to_))
        #intermediate_kf = self.bisect(from_, to_)
        intermediate_kf = self.search(from_, to_)
        print(intermediate_kf)

        #while self.input_stream.get_time() <= to_:
        for time_ in intermediate_kf:
            logging.debug('set time to {}'.format(time_))
            # use another state detector to split the 
            self.input_stream.set_pos((time_,))
            img = self.input_stream.read_ignore_lock()

            #current_hidden = len(np.argwhere(self.History.get_kf(last_kf).state != C2N('E')))
            if self.Board.hasEstimate:
                img_cropped =  self.Board.extract(img)
                cv2.imshow("backtrack", img_cropped)
                cv2.waitKey(1)
                #hidden_cnt = self.Motiondetection.getHiddenIntersectionCount(img_cropped)
                #logging.debug("Hidden intersections: {}".format(hidden_cnt))

                #if current_hidden - hidden_cnt > -2:
                logging.debug("NEW KF")
                val = self.PatchClassifier.predict(img_cropped)
                #old_state = self.Game.state.copy()
                self.Game.updateStateWithChecks(val)
                logging.debug(val)
                #new_state = self.Game.state

                self.img_overlay = self.Plot.plot_overlay(val, 
                                                    self.Board.go_board_shifted, 
                                                    img_cropped,
                                                    self.Game.manualMoves,
                                                    self.Game.last_x,
                                                    self.Game.last_y,
                                                    self.Board.border_size)
                logging.debug('Last Move:')
                logging.debug("{} {}-{}".format(N2C(self.Game.last_color), self.Game.last_x, self.Game.last_y))

                #if np.sum(new_state - old_state) != 0:
                    #pdb.set_trace()
                #last_kf = self.input_stream.get_time()
                logging.info("Delta Detected - Adding to History")
                self.update_history(val)



        # restore state and resume
        self.input_stream.set_pos((to_,))
        Signals.emit(GameRun)


    def _get_hidden_cnt(self, t):
        self.input_stream.set_pos((t,))
        img = self.input_stream.read_ignore_lock()
        img_cropped =  self.Board.extract(img)
        hidden_cnt = self.Motiondetection.getHiddenIntersectionCount(img_cropped)
        return hidden_cnt

    def search(self, from_, to_):
        lower = self._get_hidden_cnt(from_)
        upper = self._get_hidden_cnt(to_)
        kf = [] 
        for t in np.arange(from_, to_, 0.25):
            # step trough linearly in 0.2 sec steps 
            # should be fastest due to linear file acces??
            cnt = self._get_hidden_cnt(t)
            if cnt <= lower + 1:
                lower = cnt
                kf.append(t)
                print("Added {}".format(t))
        return kf



    def bisect(self, from_, to_, lower=None, upper=None):
        t_half = from_ + (to_ - from_)/2
        print('thalf: {}'.format(t_half))
        hidden_cnt = self._get_hidden_cnt(t_half)

        if lower is None:
            lower = self._get_hidden_cnt(from_)
        if upper is None:
            upper = self._get_hidden_cnt(to_)
        
        print('delta: {}'.format(to_-from_))

        if ((to_-from_) <  0.2 and
                (hidden_cnt > lower+1 or 
                 hidden_cnt > upper+1)):
            print('return []')
            # dead end we have a hand
            return []
 
        elif hidden_cnt <= lower+1:
            # intersect right
            print('Slitting right we have {} the lower bound is {}'.format(hidden_cnt, lower))
            intervals = self.bisect(t_half, to_, hidden_cnt, upper)
            return [t_half] + intervals

        #elif hidden_cnt >= upper-1:
        #    # intersect left
        #    print('Slitting left we have {} the upper bound is {}'.format(hidden_cnt, upper))
        #    intervals = self.bisect(from_, t_half, lower, hidden_cnt)
        #    return intervals + [t_half]

        else:
            # we probably have a hand on the board -> split further
            print('Slitting both we have {} the lower bound is {}, the upper bound {}'.format(hidden_cnt, lower, upper))
            intervals_lower = self.bisect(from_, t_half, lower, upper)
            intervals_upper = self.bisect(t_half, to_, lower, upper)
            return intervals_lower + intervals_upper
 


        

 
