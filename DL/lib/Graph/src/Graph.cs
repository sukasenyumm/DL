/*
* MATLAB Compiler: 4.10 (R2009a)
* Date: Sat Dec 05 14:16:35 2015
* Arguments: "-B" "macro_default" "-W" "dotnet:Graph,Graph,0.0,private" "-d"
* "D:\Graph\src" "-T" "link:lib" "-v" "class{Graph:D:\Graph2D.m}" 
*/

using System;
using System.Reflection;
using System.IO;
using MathWorks.MATLAB.NET.Arrays;
using MathWorks.MATLAB.NET.Utility;
using MathWorks.MATLAB.NET.ComponentData;
#if SHARED
[assembly: System.Reflection.AssemblyKeyFile(@"")]
#endif

namespace Graph
{
  /// <summary>
  /// The Graph class provides a CLS compliant, MWArray interface to the M-functions
  /// contained in the files:
  /// <newpara></newpara>
  /// D:\Graph2D.m
  /// <newpara></newpara>
  /// deployprint.m
  /// <newpara></newpara>
  /// printdlg.m
  /// </summary>
  /// <remarks>
  /// @Version 0.0
  /// </remarks>
  public class Graph : IDisposable
  {
      #region Constructors

      /// <summary internal= "true">
      /// The static constructor instantiates and initializes the MATLAB Component
      /// Runtime instance.
      /// </summary>
      static Graph()
      {
          if (MWMCR.MCRAppInitialized)
          {
              Assembly assembly= Assembly.GetExecutingAssembly();

              string ctfFilePath= assembly.Location;

              int lastDelimiter= ctfFilePath.LastIndexOf(@"\");

              ctfFilePath= ctfFilePath.Remove(lastDelimiter, (ctfFilePath.Length - lastDelimiter));

              string ctfFileName = MCRComponentState.MCC_Graph_name_data + ".ctf";

              Stream embeddedCtfStream = null;

              String[] resourceStrings = assembly.GetManifestResourceNames();

              foreach (String name in resourceStrings)
                {
                  if (name.Contains(ctfFileName))
                    {
                      embeddedCtfStream = assembly.GetManifestResourceStream(name);
                      break;
                    }
                }
              mcr= new MWMCR(MCRComponentState.MCC_Graph_name_data,
                             MCRComponentState.MCC_Graph_root_data,
                             MCRComponentState.MCC_Graph_public_data,
                             MCRComponentState.MCC_Graph_session_data,
                             MCRComponentState.MCC_Graph_matlabpath_data,
                             MCRComponentState.MCC_Graph_classpath_data,
                             MCRComponentState.MCC_Graph_libpath_data,
                             MCRComponentState.MCC_Graph_mcr_application_options,
                             MCRComponentState.MCC_Graph_mcr_runtime_options,
                             MCRComponentState.MCC_Graph_mcr_pref_dir,
                             MCRComponentState.MCC_Graph_set_warning_state,
                             ctfFilePath, embeddedCtfStream, true);
          }
          else
          {
              throw new ApplicationException("MWArray assembly could not be initialized");
          }
      }


      /// <summary>
      /// Constructs a new instance of the Graph class.
      /// </summary>
      public Graph()
      {
      }


      #endregion Constructors

      #region Finalize

      /// <summary internal= "true">
      /// Class destructor called by the CLR garbage collector.
      /// </summary>
      ~Graph()
      {
          Dispose(false);
      }


      /// <summary>
      /// Frees the native resources associated with this object
      /// </summary>
      public void Dispose()
      {
          Dispose(true);

          GC.SuppressFinalize(this);
      }


      /// <summary internal= "true">
      /// Internal dispose function
      /// </summary>
      protected virtual void Dispose(bool disposing)
      {
          if (!disposed)
          {
              disposed= true;

              if (disposing)
              {
                  // Free managed resources;
              }

              // Free native resources
          }
      }


      #endregion Finalize

      #region Methods

      /// <summary>
      /// Provides a void output, 0-input MWArray interface to the Graph2D M-function.
      /// </summary>
      /// <remarks>
      /// M-Documentation:
      /// Create figure
      /// </remarks>
      ///
      public void Graph2D()
      {
          mcr.EvaluateFunction(0, "Graph2D", new MWArray[]{});
      }


      /// <summary>
      /// Provides a void output, 1-input MWArray interface to the Graph2D M-function.
      /// </summary>
      /// <remarks>
      /// M-Documentation:
      /// Create figure
      /// </remarks>
      /// <param name="x">Input argument #1</param>
      ///
      public void Graph2D(MWArray x)
      {
          mcr.EvaluateFunction(0, "Graph2D", x);
      }


      /// <summary>
      /// Provides a void output, 2-input MWArray interface to the Graph2D M-function.
      /// </summary>
      /// <remarks>
      /// M-Documentation:
      /// Create figure
      /// </remarks>
      /// <param name="x">Input argument #1</param>
      /// <param name="y">Input argument #2</param>
      ///
      public void Graph2D(MWArray x, MWArray y)
      {
          mcr.EvaluateFunction(0, "Graph2D", x, y);
      }


      /// <summary>
      /// Provides a void output, 3-input MWArray interface to the Graph2D M-function.
      /// </summary>
      /// <remarks>
      /// M-Documentation:
      /// Create figure
      /// </remarks>
      /// <param name="x">Input argument #1</param>
      /// <param name="y">Input argument #2</param>
      /// <param name="titleTop">Input argument #3</param>
      ///
      public void Graph2D(MWArray x, MWArray y, MWArray titleTop)
      {
          mcr.EvaluateFunction(0, "Graph2D", x, y, titleTop);
      }


      /// <summary>
      /// Provides a void output, 4-input MWArray interface to the Graph2D M-function.
      /// </summary>
      /// <remarks>
      /// M-Documentation:
      /// Create figure
      /// </remarks>
      /// <param name="x">Input argument #1</param>
      /// <param name="y">Input argument #2</param>
      /// <param name="titleTop">Input argument #3</param>
      /// <param name="labelX">Input argument #4</param>
      ///
      public void Graph2D(MWArray x, MWArray y,
                          MWArray titleTop, MWArray labelX)
      {
          mcr.EvaluateFunction(0, "Graph2D", x, y, titleTop, labelX);
      }


      /// <summary>
      /// Provides a void output, 5-input MWArray interface to the Graph2D M-function.
      /// </summary>
      /// <remarks>
      /// M-Documentation:
      /// Create figure
      /// </remarks>
      /// <param name="x">Input argument #1</param>
      /// <param name="y">Input argument #2</param>
      /// <param name="titleTop">Input argument #3</param>
      /// <param name="labelX">Input argument #4</param>
      /// <param name="labelY">Input argument #5</param>
      ///
      public void Graph2D(MWArray x, MWArray y, MWArray titleTop,
                          MWArray labelX, MWArray labelY)
      {
          mcr.EvaluateFunction(0, "Graph2D", x, y, titleTop, labelX, labelY);
      }


      /// <summary>
      /// Provides the standard 0-input MWArray interface to the Graph2D M-function.
      /// </summary>
      /// <remarks>
      /// M-Documentation:
      /// Create figure
      /// </remarks>
      /// <param name="numArgsOut">The number of output arguments to return.</param>
      /// <returns>An Array of length "numArgsOut" containing the output
      /// arguments.</returns>
      ///
      public MWArray[] Graph2D(int numArgsOut)
      {
          return mcr.EvaluateFunction(numArgsOut, "Graph2D", new MWArray[]{});
      }


      /// <summary>
      /// Provides the standard 1-input MWArray interface to the Graph2D M-function.
      /// </summary>
      /// <remarks>
      /// M-Documentation:
      /// Create figure
      /// </remarks>
      /// <param name="numArgsOut">The number of output arguments to return.</param>
      /// <param name="x">Input argument #1</param>
      /// <returns>An Array of length "numArgsOut" containing the output
      /// arguments.</returns>
      ///
      public MWArray[] Graph2D(int numArgsOut, MWArray x)
      {
          return mcr.EvaluateFunction(numArgsOut, "Graph2D", x);
      }


      /// <summary>
      /// Provides the standard 2-input MWArray interface to the Graph2D M-function.
      /// </summary>
      /// <remarks>
      /// M-Documentation:
      /// Create figure
      /// </remarks>
      /// <param name="numArgsOut">The number of output arguments to return.</param>
      /// <param name="x">Input argument #1</param>
      /// <param name="y">Input argument #2</param>
      /// <returns>An Array of length "numArgsOut" containing the output
      /// arguments.</returns>
      ///
      public MWArray[] Graph2D(int numArgsOut, MWArray x, MWArray y)
      {
          return mcr.EvaluateFunction(numArgsOut, "Graph2D", x, y);
      }


      /// <summary>
      /// Provides the standard 3-input MWArray interface to the Graph2D M-function.
      /// </summary>
      /// <remarks>
      /// M-Documentation:
      /// Create figure
      /// </remarks>
      /// <param name="numArgsOut">The number of output arguments to return.</param>
      /// <param name="x">Input argument #1</param>
      /// <param name="y">Input argument #2</param>
      /// <param name="titleTop">Input argument #3</param>
      /// <returns>An Array of length "numArgsOut" containing the output
      /// arguments.</returns>
      ///
      public MWArray[] Graph2D(int numArgsOut, MWArray x,
                               MWArray y, MWArray titleTop)
      {
          return mcr.EvaluateFunction(numArgsOut, "Graph2D", x, y, titleTop);
      }


      /// <summary>
      /// Provides the standard 4-input MWArray interface to the Graph2D M-function.
      /// </summary>
      /// <remarks>
      /// M-Documentation:
      /// Create figure
      /// </remarks>
      /// <param name="numArgsOut">The number of output arguments to return.</param>
      /// <param name="x">Input argument #1</param>
      /// <param name="y">Input argument #2</param>
      /// <param name="titleTop">Input argument #3</param>
      /// <param name="labelX">Input argument #4</param>
      /// <returns>An Array of length "numArgsOut" containing the output
      /// arguments.</returns>
      ///
      public MWArray[] Graph2D(int numArgsOut, MWArray x, MWArray y,
                               MWArray titleTop, MWArray labelX)
      {
          return mcr.EvaluateFunction(numArgsOut, "Graph2D",
                                      x, y, titleTop, labelX);
      }


      /// <summary>
      /// Provides the standard 5-input MWArray interface to the Graph2D M-function.
      /// </summary>
      /// <remarks>
      /// M-Documentation:
      /// Create figure
      /// </remarks>
      /// <param name="numArgsOut">The number of output arguments to return.</param>
      /// <param name="x">Input argument #1</param>
      /// <param name="y">Input argument #2</param>
      /// <param name="titleTop">Input argument #3</param>
      /// <param name="labelX">Input argument #4</param>
      /// <param name="labelY">Input argument #5</param>
      /// <returns>An Array of length "numArgsOut" containing the output
      /// arguments.</returns>
      ///
      public MWArray[] Graph2D(int numArgsOut, MWArray x, MWArray y,
                               MWArray titleTop, MWArray labelX, MWArray labelY)
      {
          return mcr.EvaluateFunction(numArgsOut, "Graph2D", x, y,
                                      titleTop, labelX, labelY);
      }


      /// <summary>
      /// This method will cause a MATLAB figure window to behave as a modal dialog box.
      /// The method will not return until all the figure windows associated with this
      /// component have been closed.
      /// </summary>
      /// <remarks>
      /// An application should only call this method when required to keep the
      /// MATLAB figure window from disappearing.  Other techniques, such as calling
      /// Console.ReadLine() from the application should be considered where
      /// possible.</remarks>
      ///
      public void WaitForFiguresToDie()
      {
          mcr.WaitForFiguresToDie();
      }


      
      #endregion Methods

      #region Class Members

      private static MWMCR mcr= null;

      private bool disposed= false;

      #endregion Class Members
  }
}
