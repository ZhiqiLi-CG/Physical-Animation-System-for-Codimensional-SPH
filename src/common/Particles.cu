#include "Particles.h"
namespace ACG {

	//template class Particles<2, HOST>;
	template class Particles<3, HOST>;
	//template class Particles<2, DEVICE>;
	template class Particles<3, DEVICE>;
}