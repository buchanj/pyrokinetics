from cleverdict import CleverDict
from .decorators import not_implemented


class GKOutput(CleverDict):

    def __init__(self, *args, **kwargs):

        s_args = list(args)

        if (args and not isinstance(args[0], CleverDict)
            and isinstance(args[0], dict)):
            s_args[0] = sorted(args[0].items())

        if args:
            super(GKOutput, self).__init__(*s_args, **kwargs)
        else:
            self.default()

    def default(self):

        _data_dict = {'ntheta': 32, 'theta0': 0.0, 'nenergy': 8, 'npitch': 8, 'nky': 1, 'nkx': 1, 'kx': 0.0, 'ky': 0.1}

        super(GKOutput, self).__init__(_data_dict)

    @not_implemented
    def read_grids(self):
        """
        reads in numerical grids
        """
        pass

    @not_implemented
    def read_output_data(self,
                    ):
        """
        reads in data not currently read in by default
        """
        pass

    @not_implemented
    def read_eigenvalues(self):
        """
        reads in eigenvalue
        """
        pass

    @not_implemented
    def read_eigenfunctions(self):
        """
        reads in eigenfunction
        """
        pass

    @not_implemented
    def read_fields(self):
        """
        reads in 3D fields
        """
        pass

    @not_implemented
    def read_fluxes(self):
        """
        reads in fluxes
        """
        pass

