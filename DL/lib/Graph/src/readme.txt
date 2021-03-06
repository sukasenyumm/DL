MATLAB Builder NE (.NET Component)


1. Prerequisites for Deployment 

* Verify the MATLAB Compiler Runtime (MCR) is installed and ensure you    
  have installed version 7.10.   

* If the MCR is not installed, run MCRInstaller.exe, located in:

  C:\Program Files\MATLAB\R2009a\toolbox\compiler\deploy\win64\MCRInstaller.exe

For more information on the MCR Installer, see the MATLAB Compiler 
documentation.   
      

2. Files to Deploy and Package

Configuration                                   Files to be distributed                                                                      
============================================================================

Local component with MWArray                    Graph.dll
Local component with native API                 GraphNative.dll

* If the target machine does not have version 7.10 of 
  the MCR installed, include MCRInstaller.exe.

   

3. Resources

To learn more about:               See:
======================================================================================================
The MWArray classes                MATLAB product help or "C:\Program Files\MATLAB\R2009a\
                                   help\toolbox\dotnetbuilder\MWArrayAPI\MWArrayAPI.chm" 
Examples of .NET Web Applications  MATLAB Application Deployment 
                                   Web Example Guide


4. Definitions

MCR - MATLAB Builder NE uses the MATLAB Compiler Runtime (MCR), which is 
a standalone set of shared libraries that enable the execution of 
M-files. The MCR provides complete support for all features of MATLAB 
without the MATLAB GUI. When you package and distribute an application 
to users, you include component assemblies generated by the builder as 
well as the MATLAB Compiler Runtime (MCR). If necessary, run 
MCRInstaller.exe to install the correct version of the MCR. For more 
information about the MCR, see the MATLAB Compiler documentation.

MWArray - Use the MWArray class hierarchy, which maps to MATLAB data 
types, to pass input/output arguments to MATLAB Builder NE generated 
functions. These classes consist of a thin wrapper around a MATLAB 
array. It provides full marshaling and formatting services for all basic 
MATLAB data types including sparse arrays, structures, and cell arrays. 
These classes provide the necessary constructors, methods, and operators 
for array creation and initialization, serialization, simple indexing, 
and formatted output.

MWArray API - The MWArray API is the standard API that has been used
since the introduction of MATLAB Builder NE.  This API requires the 
MATLAB MCR to be installed on the target machine as it makes use of 
several primitive MATLAB functions. For information about using this 
API, see the MATLAB Builder NE documentation.

Native .NET API -  The Native API was designed especially, though not 
exclusively, to support .NET remoting. It allows you to pass arguments 
and return values using standard .NET types. This feature is especially 
useful for clients that need to access a remoteable component on a 
machine that does not have the MCR installed. In addition, as only 
native .NET types are used in this API, there is no need to learn 
semantics of a new set of data conversion classes. This API does not 
directly support .NET analogs for the MATLAB structure and cell array 
types. For information about using this API, see see the MATLAB Builder 
NE documentation.

.NET Framework - The Microsoft .NET Framework is a large library of 
pre-coded solutions to common programming problems that uses the 
CLR (Common Language Runtime) as the execution engine. MATLAB 
Builder NE currently supports Microsoft .NET Framework 2.0.


