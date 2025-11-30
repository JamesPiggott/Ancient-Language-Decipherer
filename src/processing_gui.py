import kivy
kivy.require('1.0.6')

from kivy.app import App
from kivy.logger import Logger
from kivy.uix.scatter import Scatter
from kivy.properties import StringProperty
from kivy.uix.button import Button


class Picture(Scatter):
    source = StringProperty(None)

class PicturesApp(App):


    def build(self):
        root = self.root
        try:
            picture = Picture(source="area_of_interest.jpg")


            btn = Button(text="Push Me !",
                         font_size="20sp",
                         background_color=(1, 1, 1, 1),
                         color=(1, 1, 1, 1),
                         size=(32, 32),
                         size_hint=(.2, .2),
                         pos=(0, 0))

            root.add_widget(btn)
            root.add_widget(picture)
        except Exception as e:
            Logger.exception('Pictures: Unable to load <%s>' % "area_of_interest.jpg")

    def on_pause(self):
        return True


if __name__ == '__main__':
    PicturesApp().run()