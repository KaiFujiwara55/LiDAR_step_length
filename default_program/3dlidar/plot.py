from typing import Any
import matplotlib.pyplot as plt

class set_plot:
    def set_ax_info(self, title=None, xlabel="X", ylabel="Y", zlabel="Z", xlim: list=None, ylim: list=None, zlim: list=None, azim=None, elev=None):
        self.title = title
        
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.zlabel = zlabel

        if xlim is not None:
            self.xlim = xlim
        
        if ylim is not None:
            self.ylim = ylim
        
        if zlim is not None:
            self.zlim = zlim


        self.azim = azim
        self.elev = elev

    def set_ax(self, ax: Any, title=None, xlabel=None, ylabel=None, zlabel=None, xlim: list=None, ylim: list=None, zlim: list=None, azim=None, elev=None, is_box_aspect=True):
        if title is not None:
            ax.set_title(title)
        elif self.title is not None:
            ax.set_title(self.title)
        
        if xlabel is not None:
            ax.set_xlabel(xlabel)
        elif self.xlabel is not None:
            ax.set_xlabel(self.xlabel)
        if ylabel is not None:
            ax.set_ylabel(ylabel)
        elif self.ylabel is not None:
            ax.set_ylabel(self.ylabel)
        
        if xlim is not None:
            ax.set_xlim(*xlim)
        elif self.xlim is not None:
            ax.set_xlim(*self.xlim)
        if ylim is not None:
            ax.set_ylim(*ylim)
        elif self.ylim is not None:
            ax.set_ylim(*self.ylim)

        if ax.name == "3d":
            if zlabel is not None:
                ax.set_zlabel(zlabel)
            elif self.zlabel is not None:
                ax.set_zlabel(self.zlabel)
            
            if zlim is not None:
                ax.set_zlim(*zlim)
            elif self.zlim is not None:
                ax.set_zlim(*self.zlim)
            
            if azim is not None:
                ax.view_init(azim=azim)
            elif self.azim is not None:
                ax.view_init(azim=self.azim)
            if elev is not None:
                ax.view_init(elev=elev)
            elif self.elev is not None:
                ax.view_init(elev=self.elev)

            if is_box_aspect:
                x = ax.get_xlim()
                y = ax.get_ylim()
                z = ax.get_zlim()
                ax.set_box_aspect([abs(x[1] - x[0]), abs(y[1] - y[0]), abs(z[1] - z[0])])
        else:
            if is_box_aspect:
                x = ax.get_xlim()
                y = ax.get_ylim()
                ax.set_box_aspect(abs(y[1] - y[0])/abs(x[1] - x[0]))

        
        return ax

